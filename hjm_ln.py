#!/usr/bin/env python
# coding: utf-8


from abc import abstractmethod
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HJM_LOG_NORMAL:
    
    """ HJM log-normal diffusion model
    
    Args:
        f0 (float): initial forward price
        alpha (1D tensor): parameter alpha
        sigma (1D tensor): parameter sigma
        ref_date (date): reference date or pricing date
        ex_dates (numpy.array): exercise dates
    """
    
    def __init__(self, f0, alpha, sigma, ref_date, ex_dates):
        
        self.f0 = f0
        self.alpha = alpha
        self.sigma = sigma
        self.ref_date = ref_date
        self.ex_dates = ex_dates
        self.diffusion_params = None
        
    @abstractmethod
    def Simulate_X(self, M):
        """Simulate state variables X (Orstein-Uhlenbeck)"""
        
        pass
    
    @abstractmethod
    def Compute_Spot_Price_From_X(self, X):
        """Compute spot price given state variables X"""
        
        pass
        
    def Simulate_Spot_Price(self, M):
        """ Simlate spot price """
        
        return self.Compute_Spot_Price_From_X(self.Simulate_X(M))


class One_Factor(HJM_LOG_NORMAL):
    
    """ One factor HJM log-normal model
    
    Args:
        Same as in the main class HJM_LOG_NORMAL
        
    """
    
    def __init__(self, f0, alpha, sigma, ref_date, ex_dates):
        super().__init__(f0, alpha, sigma, ref_date, ex_dates) 
        
        delta_tk = [(ex_dates[0] - ref_date).days] + [(d_tkp1 - d_tk).days for (d_tkp1, d_tk) in zip(ex_dates[1:], ex_dates[:-1])]
        delta_tk = (torch.tensor(delta_tk) / 365.).to(device)
        tkp1 = torch.cumsum(delta_tk, 0)
        tk = torch.cat((torch.full((1,), 0.0).to(device), tkp1), 0)[:-1]
        sigma_Z = torch.sqrt((torch.exp(2. * alpha * tkp1) - torch.exp(2. * alpha * tk)) / (2. * alpha))[:, np.newaxis]
        kappa = torch.exp(-alpha * tkp1)[:, np.newaxis]
        lamb = ((sigma ** 2) / (2. * alpha)) * (1. - torch.exp(-2. * alpha * tkp1))[:, np.newaxis]
        
        self.diffusion_params = {'kappa': kappa, 'sigma_Z': sigma_Z, 'lambda': lamb}
        
    def Simulate_X(self, M):
        N = self.diffusion_params['sigma_Z'].shape[0]
        return torch.cumsum(self.diffusion_params['sigma_Z'] * torch.randn((N, M), device = device), 0)
    
    def Compute_Spot_Price_From_X(self, X):
        return self.f0 * torch.exp(self.sigma * self.diffusion_params['kappa'] * X - 0.5 * self.diffusion_params['lambda'])



class Multi_Factor(HJM_LOG_NORMAL):
    
    """ Multi factor HJM log-normal model
    
    Args:
        Same as in the main class HJM_LOG_NORMAL +
        
        corr_mat (2D tensor): instantaneous correlation matrix
        
    """
    
    def __init__(self, f0, alpha, sigma, corr_mat, ref_date, ex_dates):
        
        self.corr_mat = corr_mat
        self.dim = len(alpha)
        super().__init__(f0, alpha, sigma, ref_date, ex_dates)
        
        N = len(ex_dates)
        
        delta_tk = [(ex_dates[0] - ref_date).days] + [(d_tkp1 - d_tk).days for (d_tkp1, d_tk) in zip(ex_dates[1:], ex_dates[:-1])]
        delta_tk = (torch.tensor(delta_tk) / 365.).to(device)
        tkp1 = torch.cumsum(delta_tk, 0)
        tk = torch.cat((torch.full((1,), 0.0).to(device), tkp1), 0)[:-1]
        
        sigma_Z = torch.zeros(N, len(alpha), len(alpha)).to(device)
        kappa = torch.zeros(len(alpha), N).to(device)
        lamb = torch.zeros(N).to(device)
    
        for idx in range(len(alpha)):
            kappa[idx] = torch.exp(-alpha[idx] * tkp1)
            lamb += ((sigma[idx] ** 2) / (2. * alpha[idx])) * (1. - torch.exp(-2. * alpha[idx] * tkp1))
        
            for idx1 in range(len(alpha)):
                sum_alpha = alpha[idx] + alpha[idx1]
                prod_sigma = sigma[idx] * sigma[idx1]
                sigma_Z[:, idx, idx1] = self.corr_mat[idx, idx1] * (torch.exp(sum_alpha * tkp1) - torch.exp(sum_alpha * tk)) / sum_alpha
            
                if idx != idx1:
                    lamb += self.corr_mat[idx, idx1] * (prod_sigma / sum_alpha) * (1. - torch.exp(-sum_alpha * tkp1))

        for idx in range(len(ex_dates)):
            sigma_Z[idx] = torch.linalg.cholesky(sigma_Z[idx], upper = True)

        self.diffusion_params = {'kappa': kappa.T[:, np.newaxis, :], 'sigma_Z': sigma_Z, 'lambda': lamb}
        
    def Simulate_X(self, M):
        N = self.diffusion_params['sigma_Z'].shape[0]
        return torch.cumsum(torch.matmul(torch.randn((N, M, self.dim), device = device), self.diffusion_params['sigma_Z']), 0)
    
    
    def Compute_Spot_Price_From_X(self, X):
        return self.f0 * torch.exp(torch.sum(self.sigma * self.diffusion_params['kappa'] * X, 2) - 0.5 * self.diffusion_params['lambda'][:, np.newaxis])

