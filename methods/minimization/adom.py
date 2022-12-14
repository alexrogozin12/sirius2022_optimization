# -*- coding: utf-8 -*-
"""adom.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OFxptJijtGvqxfOcIt5RgBXCXlLj-R71
"""

import torch
import math
import numpy as np

def Fenchel(z):
    """
    Fenchel conjugate for logloss = log(1 + exp(-yx)):
    """
    return torch.sum(torch.abs(z)*torch.log(torch.abs(z)) + (1 - torch.abs(z))*torch.log(1 - torch.abs(z)))

def dual_grad(z_):
    return torch.autograd.grad(Fenchel(z_), z_)

def adom_iteration(z, z_f, mu, alpha, nu, theta, sigma, tau, W, y):
    z_g = tau*z + (1 - tau)*z_f
    gr = dual_grad(z_g)[0]
    delta = torch.sum(sigma*W*(mu - nu*dual_grad(z_g)[0]), 1)
    mu = mu - nu*dual_grad(z_g)[0] - delta
    z = z + nu*alpha*(z_g - z) + delta
    z_f = z_g - torch.sum(theta*W*dual_grad(z_g)[0], 1)
    
    return z, z_g, z_f

def ADOM(W, mu_, L, X = None, y = None):
    """
    W - graph matrix
    x - features
    y - labels
    mu_ - strong convex constant
    L - convex constant

    stopping criterion: f(z_{k+2}) > f(z_{k+1}) > f(z_{k})
    """

    z = torch.tensor(torch.rand(10), requires_grad=True)
    mean, std = torch.mean(z), torch.std(z)
    z  = (z-mean)
    z_f = torch.empty_like(z).copy_(z)

    mu = torch.tensor(torch.from_numpy(np.full(10, np.random.random_sample())), requires_grad=True)
    mu_ = mu_
    L = L
    alpha = 1/(2*L)
    lambda_min_positive = min(np.extract(np.linalg.eigvals(W)>0, np.linalg.eigvals(W)))
    nu = torch.tensor(torch.from_numpy(np.array(2*lambda_min_positive*math.sqrt(mu_*L)/(7*max(np.linalg.eigvals(W))))), requires_grad=True)
    theta = mu_/max(np.linalg.eigvals(W))
    tau = lambda_min_positive*math.sqrt(mu_/L)/(7*max(np.linalg.eigvals(W)))
    sigma = theta/mu_

    W = W.astype(float)
    W = torch.tensor(W, requires_grad = True)
    F = []
    i = 0
    while(i != -1):
        z, z_g, z_f = adom_iteration(z, z_f, mu, alpha, nu, theta, sigma, tau, W, y)
        F.append(Fenchel(z_g))
        if ((i > 2) and ((F[i] > F[i-1]) and (F[i-1] > F[i-2]))): i = -1
        else: i+=1
    return z, z_g, z_f, F