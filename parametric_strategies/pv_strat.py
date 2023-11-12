#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from tools import*
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Compute_Swing_Price(model, optimizer_configs, grid, strike_price, n_pack_eval, pack_size_eval, output_delta = False):
    params_training = Training(model, optimizer_configs, grid, strike_price)
    return valuation(model, params_training, strike_price, grid, n_pack_eval, pack_size_eval, output_delta)



def Training(model, optimizer_configs, grid, strike_price):
    
    """ Learning phase for pv_strat
    
    Args:
        model (class): a diffusion model
        optimizer_configs (dict): optimizer configurations
        grid (class): swing volume grid
        strike_price (float): fixed strike price
    """
        
    n_dates = len(grid.ex_dates)
    
    # pretraining in case of transfer learning
    if optimizer_configs['transfer_learning']['activate']:
        if n_dates <= 31:
            raise Exception("mid month aggregation only available for at least 32 exercise dates !")
            
        else:
            agg_grid, len_agg_periods = grid.Compute_agregated_grid()
            optimizer_configs['transfer_learning']['activate'] = False
            optimizer_configs['n_iterations'], optimizer_configs['transfer_learning']['n_iter_pre_training'] = optimizer_configs['transfer_learning']['n_iter_pre_training'], optimizer_configs['n_iterations']
            print(">---------------------- Pretraining ----------------------<")
            pre_training_params = Training_pv_strat(model, optimizer_configs, agg_grid, strike_price)
            print(">---------------------- Training ----------------------<")
            optimizer_configs['transfer_learning']['activate'] = True
            optimizer_configs['n_iterations'], optimizer_configs['transfer_learning']['n_iter_pre_training'] = optimizer_configs['transfer_learning']['n_iter_pre_training'], optimizer_configs['n_iterations']

            params_init = np.full((n_dates, 3), 0.0)
            j = 0

            for i in range(len(agg_grid.ex_dates)):
                n_days = len_agg_periods[i]
                params_init[j : j + n_days] = pre_training_params[i].detach().cpu().numpy()
                j += n_days
    
    params = params_init if optimizer_configs['transfer_learning']['activate'] else np.random.randn(len(grid.ex_dates), 3)
    params = torch.tensor(params, device = device, requires_grad = True)
    optimizers = Init_Optimizer([params], optimizer_configs)
    norm_Q = grid.Q_max if grid.Q_max == grid.Q_min else grid.Q_max - grid.Q_min
    start_time = time.time()
    
    ##############################################################################################
    #######################################               TRAINING
    
    for epoch in range(optimizer_configs['n_iterations']):
        
        for batch in range(optimizer_configs['nb_batches']):
            psi = model.Simulate_Spot_Price(optimizer_configs['batch_size']) - strike_price
            Q = torch.zeros_like(psi[0])
            running_cf = torch.zeros_like(psi[0])
            optimizers.zero_grad()
                    
            for n in range(n_dates):
                margin_Q = (Q - grid.Q_min) / norm_Q
                xi = nn.Sigmoid()(params[n][0] * psi[n] + margin_Q * params[n][1] + params[n][2])
                q = grid.Compute_Constrained_Control(xi, Q, n)
                Q += q
                running_cf += q * psi[n]
    
            loss = -torch.mean(running_cf)
            loss.backward()
            optimizers.step()
            
    exc_time = round(time.time() - start_time, 2)
    print("--- Training time (pv_strat): %s seconds ---" % (exc_time))

    return params


def valuation(model, params_training, strike_price, grid, n_pack, pack_size, output_delta):
    
    """ Evaluation phase for nn_strat
    
    Args:
        model (class): a diffusion model
        params_training (tensor): trained parameters for pv_strat
        strike_price (float): fixed strike price
        grid (class): swing volume grid
        n_pack (int): number of packs to be generated for a sequential valuation
        pack_size (int): size of each pack
        output_delta (bool): boolean to compute sensitivity of the swing price with respect to the initial forward price f0
    """
    
    swing_price_test = 0.0
    swing_price_test_bang_bang = 0.0
    n_dates = len(grid.ex_dates)
    var_price = 0.0
    var_price_bang_bang = 0.0
    M = float(n_pack * pack_size) ** 0.5
    
    if output_delta:
        delta = np.full(n_dates, 0.0)
        delta_bang_bang = np.full(n_dates, 0.0)
        
    norm_Q = grid.Q_max if grid.Q_max == grid.Q_min else grid.Q_max - grid.Q_min
    
    with torch.no_grad():
            
        # compute the estimator for each pack
        for idx in range(n_pack):
            psi_test = model.Simulate_Spot_Price(pack_size) - strike_price
            Q_test = torch.zeros_like(psi_test[0])
            Q_test_bang_bang = torch.zeros_like(psi_test[0])
            running_cf_test = torch.zeros_like(psi_test[0])
            running_cf_test_bang_bang = torch.zeros_like(psi_test[0])
    
            for n in range(n_dates):
                margin_Q_test = (Q_test - grid.Q_min) / norm_Q
                margin_Q_test_bang_bang = (Q_test_bang_bang - grid.Q_min) / norm_Q
                xi_test = nn.Sigmoid()(params_training[n][0] * psi_test[n] + margin_Q_test * params_training[n][1] + params_training[n][2])
                xi_test_bang_bang = nn.Sigmoid()(params_training[n][0] * psi_test[n] + margin_Q_test_bang_bang * params_training[n][1] + params_training[n][2])
                xi_test_bang_bang = (xi_test_bang_bang >= 0.5).float() # shrinkage for bang bang
                q_test = grid.Compute_Constrained_Control(xi_test, Q_test, n)
                q_test_bang_bang = grid.Compute_Constrained_Control(xi_test_bang_bang, Q_test_bang_bang, n)
                Q_test += q_test
                Q_test_bang_bang += q_test_bang_bang
                running_cf_test += q_test * psi_test[n]
                running_cf_test_bang_bang += q_test_bang_bang * psi_test[n]
                
                if output_delta:
                    delta[n] += torch.mean(q_test * (psi_test[n] + strike_price) / model.f0).item()
                    delta_bang_bang[n] += torch.mean(q_test_bang_bang * (psi_test[n] + strike_price) / model.f0).item()
                
            swing_price_test += torch.mean(running_cf_test).item()
            swing_price_test_bang_bang += torch.mean(running_cf_test_bang_bang).item()
                
            if grid.normalized_swing:
                delta_loc_const = grid.q_max_old[0] - grid.q_min_old[0]
                swing_price_test = swing_price_test * delta_loc_const + grid.q_min_old[0] * n_dates * (model.f0 - strike_price)
                swing_price_test_bang_bang = swing_price_test_bang_bang * delta_loc_const + grid.q_min_old[0] * n_dates * (model.f0 - strike_price)
                
                if output_delta:
                    to_do = "add formula for delta with swing normalization"
                
            var_price += torch.sum(torch.pow(running_cf_test / M, 2.0)).item()
            var_price_bang_bang += torch.sum(torch.pow(running_cf_test_bang_bang / M, 2.0)).item()
           
        # average of estimators obtained in each pack
        swing_price_test /= n_pack
        swing_price_test_bang_bang /= n_pack
        var_price -= swing_price_test ** 2
        var_price_bang_bang -= swing_price_test_bang_bang ** 2
        
        if output_delta:
            delta /= n_pack
            delta_bang_bang /= n_pack
            
            return {'swing_price' : round(swing_price_test, 2), 'var_price': round(var_price, 2), 'swing_price_bang_bang': round(swing_price_test_bang_bang, 2), 
                    'var_price_bang_bang': round(var_price_bang_bang, 2), 'delta_fwd': delta, 'delta_fwd_bg': delta_bang_bang}
        
        else:
            return {'swing_price' : round(swing_price_test, 2), 'var_price': round(var_price, 2), 'swing_price_bang_bang': round(swing_price_test_bang_bang, 2), 
                    'var_price_bang_bang': round(var_price_bang_bang, 2)}

