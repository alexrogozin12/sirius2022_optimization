import numpy as np

from typing import List, Optional
from oracles.minimization import BaseSmoothOracle, OracleLinearComb
from .base import BaseDecentralizedMethod
from .logger import LoggerDecentralized
from scipy.stats import ortho_group


class DAPDG(BaseDecentralizedMethod):
    
    """
    oracle_list: list of size M of oracles for every node
    x_0: matrix of shape [M, N], must be if Range A^T
    B: matrix of affine constraints of shape [P, N]
    W: laplasian matrix of graph of shape [M, M]
    logger: haven't figured it out yet
    eta, alpha, beta, theta, sigma, gamma: parameters
    """
    def __init__(
            self,
            oracle_list: tp.List[BaseSmoothOracle],
            x_0: np.ndarray,
            B: np.ndarray,
            W: np.ndarray,
            logger: LoggerDecentralized,
            gamma: float = 0.75
    ):
        super().__init__(oracle_list, x_0, logger)
                
        self.B = B
        self.W = W
        
        M = W.shape[0]
        P, D = B.shape
        self.D, self.M = D, M
        
        assert W.shape == (M, M)
        assert self.x.shape == (M, D)
        
        self.gamma = gamma  # parameter for matrix A

        bold_B = np.kron(np.ones((M, M)), B)
        bold_W = np.kron(W, np.ones((D, D)))
        self.A = np.vstack([bold_B, gamma * bold_W])
        
        L_mu_params = self.compute_l_mu(oracle_list, self.A)
        self.init_parameters(**L_mu_params)
        assert self.eta > 0 and self.alpha > 0 and self.beta > 0
        assert 0 < self.tau <= 1 and 0 < self.sigma <= 1
        
        self.x_f = np.zeros((M, D))
    
        
    def compute_l_mu(self, oracle_list, mat):
        oracle = OracleLinearComb(oracle_list, [1 / len(oracle_list) for _ in oracle_list])
        l_x = max([np.linalg.svd(oracle.A)[1].max() / oracle.den + oracle.regcoef for oracle in oracle_list])
        mu_x = min([oracle.regcoef for oracle in oracle_list])
        sing_vals = np.linalg.svd(mat)[1]
        l_xy = sing_vals.max()**2
        mu_xy = sing_vals[sing_vals >= 1e-10].min()**2
        return dict(L_x=l_x, mu_x=mu_x, L_xy=l_xy, mu_xy=mu_xy)
        
    
    def init_parameters(self, mu_x, mu_xy, L_x, L_xy):
        delta = (mu_xy**2 / (2*mu_x*L_x))**.5
        rho_b = (4 + 8 * max(L_xy / mu_xy * (L_x / mu_x)**.5,
                             L_xy**2 / mu_xy**2)) ** (-1)
        self.sigma = (mu_x / (2 * L_x))**.5
        self.alpha = mu_x
        # self.theta = 1 - rho_b
        self.eta = min(1 / (4 * (mu_x + L_x * self.sigma)), delta / (4 * L_xy))
        self.beta = 1 / (2 * self.eta * L_xy ** 2)
        self.tau = (self.sigma**(-1) + 0.5)**(-1)
        

    def step(self):
        A = self.A
        x = self.x.reshape(self.M * self.D)
        x_f = self.x_f.reshape(self.M * self.D)
        x_g = self.tau * x + (1 - self.tau) * x_f
        
        grad_F_x_g = self.grad_list(x_g.reshape(M, D)).reshape(D * M)  # np.array[D * M]
        
        x_new = x + self.eta * self.alpha * (x_g - x) - \
            self.eta * (self.beta * A.T.dot(A.dot(x)) - grad_F_x_g)
        
        x_f = x_g + self.sigma * (x_new - x)
        
        self.x = x_new.reshape(self.M, self.D)
        self.x_f = x_f.reshape(self.M, self.D)
        
        
    """
    W: laplasian matrix of graph of shape [M, M]
    """
    def set_new_graph(self, W):
        assert W.shape == (self.M, self.M)
        bold_B = np.kron(np.ones((self.M, self.M)), self.B)
        bold_W = np.kron(W, np.ones((self.N, self.N)))
        self.A = np.vstack([bold_B, self.gamma * bold_W])
