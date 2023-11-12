#!/usr/bin/env python
# coding: utf-8

from langevin_optimizer import*


def Init_Optimizer(params, optimizer_configs):
    
    """ Initialization of the optimizer
    
    Args:
        params (list): parameters to be optimized
        optimizer_configs (dict): optimizer configurations
    """
    
    if optimizer_configs["algo_name"] == "Adam":
        return torch.optim.Adam(params, lr = optimizer_configs["update_infos"]["lr"], amsgrad = optimizer_configs["update_infos"]["amsgrad"])
        
    elif optimizer_configs["algo_name"] == "pSGLD":
        return pSGLD(params, lr = optimizer_configs["update_infos"]["lr"], sigma = optimizer_configs["update_infos"]["sigma"], 
                     beta = optimizer_configs["update_infos"]["beta"], Lambda = optimizer_configs["update_infos"]["lambda"], weight_decay=0, centered=False)
        
    else:
        raise Exception("Selected optimizer not managed !")

