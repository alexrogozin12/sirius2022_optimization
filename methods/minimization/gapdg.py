import numpy as np

from typing import List, Optional
from oracles.minimization import BaseSmoothOracle, OracleLinearComb
from .base import BaseDecentralizedMethod
from .logger import LoggerDecentralized


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
            oracle_list: List[BaseSmoothOracle],
            x_0: np.ndarray,
            B: np.ndarray,
            W: np.ndarray,
            logger: LoggerDecentralized,
            eta: float = 1.0,
            alpha: float = 0.75,
            beta: float = 0.01,
            theta: float = 0.01,
            sigma: float = 0.01,
            gamma: float = 0.75
    ):
        super().__init__(oracle_list, x_0, logger)
        
        assert eta > 0 and alpha > 0 and beta > 0
        assert 0 < theta <= 1 and 0 < sigma <= 1
        
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.sigma = sigma
        self.gamma = gamma  # parameter for matrix A
        
        self.B = B
        self.W = W
        
        M = W.shape[0]
        N = B.shape[1]
        self.N, self.M = N, M
        
        assert W.shape == (M, M)
        assert self.x.shape == (M, N)
        
        self.x_f = np.zeros((M, N))
        
        bold_B = np.kron(np.ones((M, M)), B)
        bold_W = np.kron(W, np.ones((N, N)))
        self.A = np.vstack([bold_B, gamma * bold_W])
        
        # x = np.random.rand(N * M)
        # self.x = A.T.dot(x)


    def step(self):
        A = self.A
        x = self.x.reshape(M * N)
        x_f = self.x_f.reshape(M * N)
        x_g = self.theta * x + (1 - self.theta) * x_f
        
        grad_F_x_g = self.grad_list(x_g.reshape(M, N)).reshape(N * M)  # np.array[N * M]
        
        x_new = x + self.eta * self.alpha * (x_g - x) - \
            self.eta * (self.beta * A.T.dot(A.dot(x)) - grad_F_x_g)
        
        x_f = x_g + self.sigma * (x_new - x)
        
        self.x = x_new.reshape(M, N)
        self.x_f = x_f.reshape(M, N)
        
        
    """
    W: laplasian matrix of graph of shape [M, M]
    """
    def set_new_graph(self, W):
        assert W.shape == (self.M, self.M)
        bold_B = np.kron(np.ones((self.M, self.M)), self.B)
        bold_W = np.kron(W, np.ones((self.N, self.N)))
        self.A = np.vstack([bold_B, self.gamma * bold_W])