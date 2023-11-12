#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from tools import*
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Compute_Swing_Price(model, nn_hidden_sizes, optimizer_configs, grid, strike_price, n_pack_eval, pack_size_eval, output_delta = False):
    nn_training = Training(model, nn_hidden_sizes, optimizer_configs, grid, strike_price)
    return valuation(model, nn_training, grid, strike_price, n_pack_eval, pack_size_eval, output_delta)



def Training(model, nn_hidden_sizes, optimizer_configs, grid, strike_price):
    
    """ Learning phase for nn_strat
    
    Args:
        model (class): a diffusion model
        nn_hidden_sizes (array): number of unit per hidden layer
        optimizer_configs (dict): optimizer configurations
        grid (class): swing volume grid
        strike_price (float): fixed strike price
    """
    
    ##############################################################################################
    #######################################               MODELS INIT

    # init feedforward neural network
    n_ex_dates = len(grid.ex_dates)
    modules = []
    modules.append(nn.Linear(3, nn_hidden_sizes[0]))
    modules.append(nn.BatchNorm1d(nn_hidden_sizes[0]))
    modules.append(nn.ReLU())
    
    for i in range(1, len(nn_hidden_sizes)):
        modules.append(nn.Linear(nn_hidden_sizes[i - 1], nn_hidden_sizes[i]))
        modules.append(nn.BatchNorm1d(nn_hidden_sizes[i]))
        modules.append(nn.ReLU())
        
    modules.append(nn.Linear(nn_hidden_sizes[-1], 3))
    modules.append(nn.BatchNorm1d(3))
    nn_models = nn.Sequential(*modules).to(device)
    
    # pretraining in case of transfer learning
    if optimizer_configs['transfer_learning']['activate']:
        if n_ex_dates <= 31:
            raise Exception("mid month aggregation only available for at least 32 exercise dates !")
            
        else:
            agg_grid, _ = grid.Compute_agregated_grid()
            optimizer_configs['transfer_learning']['activate'] = False
            optimizer_configs['n_iterations'], optimizer_configs['transfer_learning']['n_iter_pre_training'] = optimizer_configs['transfer_learning']['n_iter_pre_training'], optimizer_configs['n_iterations']
            print(">---------------------- Pretraining ----------------------<")
            pre_training_params = Training_nn_strat(model, nn_hidden_sizes, optimizer_configs, agg_grid, strike_price)
            print(">---------------------- Training ----------------------<")
            nn_models.load_state_dict(pre_training_params.state_dict())
            optimizer_configs['transfer_learning']['activate'] = True
            optimizer_configs['n_iterations'], optimizer_configs['transfer_learning']['n_iter_pre_training'] = optimizer_configs['transfer_learning']['n_iter_pre_training'], optimizer_configs['n_iterations']
    
    optimizers = Init_Optimizer(nn_models.parameters(), optimizer_configs)
    norm_Q = grid.Q_max if grid.Q_max == grid.Q_min else grid.Q_max - grid.Q_min
        
    ##############################################################################################
    #######################################               TRAINING
    
    start_time = time.time()
    
    for epoch in range(optimizer_configs['n_iterations']):
        
        for batch in range(optimizer_configs['nb_batches']):
            optimizers.zero_grad()
            psi = model.Simulate_Spot_Price(optimizer_configs['batch_size']) - strike_price
            Q = torch.zeros_like(psi[0][:, np.newaxis])
            running_cf = torch.zeros_like(Q)
            unit_tensor = torch.ones_like(Q)
                                    
            for n in range(n_ex_dates):
                time_input = ((grid.ex_dates[n] - model.ref_date).days / 365.0) * unit_tensor
                margin_Q = (Q - grid.Q_min) / norm_Q
                inputs = torch.cat((time_input, psi[n][:, np.newaxis], margin_Q), 1).float()
                nn_outputs = nn_models(inputs)
                xi = nn.Sigmoid()(torch.sum(nn_outputs * inputs, 1))[:, np.newaxis]
                q = grid.Compute_Constrained_Control(xi, Q, n)
                Q += q
                running_cf += q * psi[n][:, np.newaxis]
            
            loss = -torch.mean(running_cf)
            loss.backward()
            optimizers.step()
            
    exc_time = round(time.time() - start_time, 2)
    print("--- Training time (nn_strat): %s seconds ---" % (exc_time))
 
    return nn_models


def valuation(model, nn_model_training, grid, strike_price, n_pack, pack_size, output_delta):  
    
    """ Evaluation phase for nn_strat
    
    Args:
        model (class): a diffusion model
        nn_model_training (module): trained feedforward neural network
        grid (class): swing volume grid
        strike_price (float): fixed strike price
        n_pack (int): number of packs to be generated for a sequential valuation
        pack_size (int): size of each pack
        output_delta (bool): boolean to compute sensitivity of the swing price with respect to the initial forward price f0
    """

    price = 0.0
    price_bang_bang = 0.0
    var_price = 0.0
    var_price_bang_bang = 0.0
    M = float(n_pack * pack_size) ** 0.5
    n_dates = len(grid.ex_dates)
    
    if output_delta:
        delta = np.full(n_dates, 0.0)
        delta_bang_bang = np.full(n_dates, 0.0)
        
    norm_Q = grid.Q_max if grid.Q_max == grid.Q_min else grid.Q_max - grid.Q_min
    
    with torch.no_grad():
                
        # compute the estimator for each pack
        for idx in range(n_pack):
            psi_test = model.Simulate_Spot_Price(pack_size) - strike_price
            Q_test = torch.zeros_like(psi_test[0][:, np.newaxis])
            cf = torch.zeros_like(Q_test)
            Q_test_bang_bang = torch.zeros_like(Q_test)
            cf_bang_bang = torch.zeros_like(Q_test)
            unit_tensor = torch.ones_like(Q_test)
                
            for n in range(n_dates):
                time_input = ((grid.ex_dates[n] - model.ref_date).days / 365.0) * unit_tensor
                margin_Q_test = (Q_test - grid.Q_min) / norm_Q
                margin_Q_test_bang_bang = (Q_test_bang_bang - grid.Q_min) / norm_Q
                input_test = torch.cat((time_input, psi_test[n][:, np.newaxis], margin_Q_test), 1).float()
                nn_outputs_test = nn_model_training(input_test)
                input_test_bang_bang = torch.cat((time_input, psi_test[n][:, np.newaxis], margin_Q_test_bang_bang), 1)
                nn_outputs_test_bang_bang = nn_model_training(input_test_bang_bang)
                xi_test = nn.Sigmoid()(torch.sum(nn_outputs_test * input_test, 1))[:, np.newaxis]
                xi_test_bang_bang = nn.Sigmoid()(torch.sum(nn_outputs_test_bang_bang * input_test_bang_bang, 1))[:, np.newaxis]
                xi_test_bang_bang = (xi_test_bang_bang >= 0.5).float() # shrinkage for bang bang
                q_test = grid.Compute_Constrained_Control(xi_test, Q_test, n)
                q_test_bang_bang = grid.Compute_Constrained_Control(xi_test_bang_bang, Q_test_bang_bang, n)
                Q_test += q_test
                Q_test_bang_bang += q_test_bang_bang
                cf += q_test * psi_test[n][:, np.newaxis]
                cf_bang_bang += q_test_bang_bang * psi_test[n][:, np.newaxis]

                if output_delta:
                    delta[n] += torch.mean(q_test * (psi_test[n] + strike_price) / model.f0).item()
                    delta_bang_bang[n] += torch.mean(q_test_bang_bang * (psi_test[n] + strike_price) / model.f0).item()
                    
            price += torch.mean(cf).item()
            price_bang_bang += torch.mean(cf_bang_bang).item()
                
            if grid.normalized_swing:
                delta_loc_const = grid.q_max_old[0] - grid.q_min_old[0]
                price = price * delta_loc_const + grid.q_min_old[0] * n_dates * (model.f0 - strike_price)
                price_bang_bang = price_bang_bang * delta_loc_const + grid.q_min_old[0] * n_dates * (model.f0 - strike_price)
                
                if output_delta:
                    to_do = "add formula for delta with swing normalization"
                    
            var_price += torch.sum(torch.pow(cf / M, 2.0)).item()
            var_price_bang_bang += torch.sum(torch.pow(cf_bang_bang / M, 2.0)).item()
            
        # average of estimators obtained in each pack
        price /= n_pack
        price_bang_bang /= n_pack
        var_price -= price ** 2
        var_price_bang_bang -= price_bang_bang ** 2
        
        if output_delta:
            delta /= n_pack
            delta_bang_bang /= n_pack
            
            return {'swing_price' : round(price, 2), 'var_price': round(var_price, 2), 'swing_price_bang_bang': round(price_bang_bang, 2), 
                    'var_price_bang_bang': round(var_price_bang_bang, 2), 'delta_fwd': delta, 'delta_fwd_bg': delta_bang_bang}
        
        else:
            return {'swing_price' : round(price, 2), 'var_price': round(var_price, 2), 'swing_price_bang_bang': round(price_bang_bang, 2), 
                    'var_price_bang_bang': round(var_price_bang_bang, 2)}

