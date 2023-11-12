#!/usr/bin/env python
# coding: utf-8


import calendar
import torch
from dateutil.relativedelta import relativedelta



class Volume_Grid:
    
    """ Swing volume grid management
    
    Args:
        Q_min (float): global constraint min
        Q_max (float): global constraint max
        q_min (1D tensor): local constraint min
        q_max (1D tensor): local constraint max
        ex_dates (numpy.array): exercise dates
        normalized_swing (bool): a boolean flag to normalize the swing contract with local contracts becoming q_min = 0 and q_max = 1
    """
    
    def __init__(self, Q_min, Q_max, q_min, q_max, ex_dates, normalized_swing = False):
        
        self.Q_min = Q_min 
        self.Q_max = Q_max
        self.ex_dates = ex_dates
        self.q_min = q_min
        self.q_max = q_max
        self.normalized_swing = normalized_swing
        
        zeros_tensor = torch.zeros(len(q_min))
        unit_tensor = torch.ones(len(q_min))
        n_ex_dates = len(ex_dates)
        
        if normalized_swing:
            self.q_min_old = self.q_min
            self.q_min = torch.zeros_like(q_min)
            self.q_max_old = self.q_max
            self.q_max = torch.ones_like(q_max)
            self.Q_min = nn.ReLU()(Q_min - n_ex_dates * torch.sum(q_min)) / (q_max[0] - q_min[0])
            self.Q_max = nn.ReLU()(Q_max - n_ex_dates * torch.sum(q_min)) / (q_max[0] - q_min[0])
            
        # Upper Bound Volume Grid
        self.upper_bound = torch.zeros(n_ex_dates + 1)
        self.upper_bound[0] = 0.0
        self.upper_bound[n_ex_dates] = self.Q_max
        self.upper_bound[1 : n_ex_dates] = torch.minimum(torch.cumsum(self.q_max, 0)[:-1], self.Q_max * unit_tensor[:-1])
        
        # Lower Bound Volume Grid
        self.lower_bound = torch.zeros(n_ex_dates + 1)
        self.lower_bound[n_ex_dates] = self.Q_min
        self.lower_bound[0] = 0.0        
        self.lower_bound[1 : n_ex_dates] = torch.flip(torch.maximum(zeros_tensor[:-1], self.Q_min - torch.cumsum(torch.flip(self.q_max, (0,)) , 0)[:-1]), (0,))
        
        
    def Compute_agregated_grid(self):
        """ Compute an aggregated swing volume grid. Useful for transfer learning"""
        
        n_ex_dates = len(self.ex_dates)
        
        if n_ex_dates < 60: # no aggregation for contracts with less than 2 months
            raise Exception("Cannot agregate !")
            
        else:
            last_ex_date = self.ex_dates[n_ex_dates - 1]
            today = self.ex_dates[0]
            actual_n_ex_dates = (last_ex_date.year - today.year) * 12 + (last_ex_date.month - today.month) + 1
            agg_ex_dates = []
            new_q_min = torch.zeros(actual_n_ex_dates)
            new_q_max = torch.zeros(actual_n_ex_dates)
            
            for i in range(actual_n_ex_dates):
                last_date_period = min(datetime(today.year, today.month, calendar.monthrange(today.year,today.month)[1]), last_ex_date)
                mid_period = (last_date_period - today).days // 2
                agg_ex_dates.append(datetime(today.year, today.month, mid_period))
                today = today + relativedelta(months=1)
                today = datetime(today.year, today.month, 1)
                
            cum_n_days_in_months = 0
            starting_date_index = 0
            today = self.ex_dates[0]
            len_agg_periods = np.full(actual_n_ex_dates, 0)
        
            for i in range(0, actual_n_ex_dates):
                if today.month == last_ex_date.month and today.year == last_ex_date.year:
                    cum_n_days_in_months+= (last_ex_date - datetime(today.year, today.month, 1)).days + 1
                
                else:
                    cum_n_days_in_months += calendar.monthrange(today.year,today.month)[1] - today.day + 1
                    
                len_agg_periods[i] = cum_n_days_in_months
                new_q_max[i] = torch.sum(self.q_max[starting_date_index : cum_n_days_in_months])
                new_q_min[i] = torch.sum(self.q_min[starting_date_index : cum_n_days_in_months])
                starting_date_index = cum_n_days_in_months
                today = today + relativedelta(months = 1)
                today = datetime(today.year, today.month, 1)
            
            return Volume_Grid(self.Q_min, self.Q_max, new_q_min, new_q_max, agg_ex_dates), len_agg_periods
        
    
    def Compute_Constrained_Control(self, xi, Q, n):
        """Compute admissible volume to be purchased given a cumulative consumption up to date n"""
        
        A1 = torch.maximum(self.lower_bound[n + 1] - Q, self.q_min[n])
        A2 = torch.minimum(self.upper_bound[n + 1] - Q, self.q_max[n])
        
        return A1 + (A2 - A1) * xi

