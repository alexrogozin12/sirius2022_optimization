import random
import numpy as np
import matplotlib.pyplot as plt

def calc_abs_value(X):
    return np.sqrt((X**2).sum())

def print_log(i, X, S, abs_delta_Fx):
    print(i, "\tX:\t", np.transpose(X))
    print(i, "\tS:\t", np.transpose(S))
    print('sum delta X:\t', abs_delta_Fx)
    print()
    
def make_graph(abs_delta_Fx):
    iterations = [i for i in range(len(abs_delta_Fx))]
    #plt.plot(iterations, abs_delta_Fx)
    plt.plot(iterations, abs_delta_Fx, '--bo')
    plt.show()

def logres_fun(x):
    return 1/(1+np.exp(x))
    
def calc_next_x_1(W, X, alpha, S):
    return W * X - np.multiply(S, alpha)

def calc_curr_s_1(W, S, X_curr, X_prev, calc_delta_Fx):
    return W * S  + calc_delta_Fx(W, X_curr) - calc_delta_Fx(W, X_prev)
    
def calc_next_x_2(W, Y, alpha, S):
    return W * Y - np.multiply(S, alpha)

def calc_curr_s_2(W, S, Y_curr, Y_prev, calc_delta_Fx):
    return W * S  + calc_delta_Fx(W, Y_curr) - calc_delta_Fx(W, Y_prev)

def calc_next_z_2(W, Z, S, alpha, theta):
    return W * Z - (alpha / theta) * S

def calc_curr_y_2(Z, X, theta):
    return np.multiply(Z, theta) + np.multiply(X, (1 - theta))
    
def calc_next_x_3(W, Y, alpha, S):
    return W * Y - np.multiply(S, alpha)

def calc_curr_s_3(W, S, Y_curr, Y_prev, calc_delta_Fx):
    return W * S  + calc_delta_Fx(W, Y_curr) - calc_delta_Fx(W, Y_prev)

def calc_next_z_3(W, Y, Z, S, alpha, theta, mu):
    tmp_val = W * (np.multiply(Y, mu*alpha/theta) + Z) - (alpha / theta) * S
    return W * Z - (alpha / theta) * S

def calc_curr_y_3(Z, X, theta):
    return np.multiply(Z, theta) + np.multiply(X, (1 - theta))

def gradient_tracking_1(W, X0, e, alpha, calc_delta_Fx,
        calc_next_x = calc_next_x_1, 
        calc_curr_s = calc_curr_s_1,
        max_iter=100,
        use_logging=False,
        need_graph=False):

    S0 = calc_delta_Fx(W, X0)
    X = calc_next_x(W, X0, alpha, S0)
    
    i = 0
    abs_delta_Fx = [calc_abs_value(calc_delta_Fx(W, X))]
    if use_logging: print_log(i, X, S0, abs_delta_Fx[-1])
    
    while abs_delta_Fx[-1] > e:
        X0 = X
        S = calc_curr_s(W, S0, X, X0, calc_delta_Fx)
        X = calc_next_x(W, X, alpha, S)
        abs_delta_Fx.append(calc_abs_value(calc_delta_Fx(W, X)))
        
        i += 1
        if use_logging: print_log(i, X, S, abs_delta_Fx[-1])
        if i > max_iter: break
    if make_graph: make_graph(abs_delta_Fx)
    return X

def gradient_tracking_2(W, X0, e, alpha, theta, calc_delta_Fx, 
        calc_next_x = calc_next_x_2, 
        calc_curr_y = calc_curr_y_2, 
        calc_next_z = calc_next_z_2, 
        calc_curr_s = calc_curr_s_2,
        max_iter=100,
        use_logging=False,
        need_graph=False):

    Y0 = Z0 = X0
    S0 = calc_delta_Fx(W, X0)

    Y = calc_curr_y(Z0, X0, theta)
    S = calc_curr_s(W, S0, Y, Y0, calc_delta_Fx)
    X = calc_next_x(W, X0, alpha, S)
    Z = calc_next_z(W, Z0, S, alpha, theta)

    i = 0
    abs_delta_Fx = [calc_abs_value(calc_delta_Fx(W, X))]
    if use_logging: print_log(i, X, S, abs_delta_Fx[-1])
    while abs_delta_Fx[-1] > e:
        Y0 = Y
        Y  = calc_curr_y(Z, X, theta)
        S  = calc_curr_s(W, S0, Y, Y0, calc_delta_Fx)
        X  = calc_next_x(W, X, alpha, S)
        Z  = calc_next_z(W, Z, S, alpha, theta)
        abs_delta_Fx.append(calc_abs_value(calc_delta_Fx(W, X)[-1]))

        i+=1
        if use_logging: print_log(i, X, S, abs_delta_Fx[-1])
        if i > max_iter: break
    if need_graph: make_graph(abs_delta_Fx)
    return X

def gradient_tracking_3(W, X0, e, alpha, theta, mu, calc_delta_Fx,
        calc_next_x = calc_next_x_3, 
        calc_curr_y = calc_curr_y_3, 
        calc_next_z = calc_next_z_3, 
        calc_curr_s = calc_curr_s_3,
        max_iter=100,
        use_logging=False,
        need_graph=False):

    Y0 = Z0 = X0
    S0 = calc_delta_Fx_3(W, X0)

    Y = calc_curr_y_3(Z0, X0, theta)
    S = calc_curr_s_3(W, S0, Y, Y0, calc_delta_Fx)
    X = np.multiply(Z0, theta) - np.multiply(W*X0, (1 - theta))
    Z = W * Z0 - (alpha / ( theta + mu * alpha) ) * S

    i = 0
    abs_delta_Fx = [calc_abs_value(calc_delta_Fx_3(W, X))]
    if use_logging: print_log(i, X, S, abs_delta_Fx)
    while abs_delta_Fx[-1] > e:
        Y0 = Y
        Y  = calc_curr_y_3(Z, X, theta)
        S  = calc_curr_s_3(W, S0, Y, Y0, calc_delta_Fx)
        X  = calc_next_x_3(W, X, alpha, S)
        Z  = calc_next_z_3(W, Y, Z, S, alpha, theta, mu)
        abs_delta_Fx.append(calc_abs_value(calc_delta_Fx_3(W, X)[-1]))

        i+=1
        if use_logging: print_log(i, X, S, abs_delta_Fx[-1])
        if i > max_iter: break
    if need_graph: make_graph(abs_delta_Fx)
    return X

def gradient_tracking(W, X0, e, alpha, theta, mu, calc_delta_Fx, max_iter=100, use_logging=False, need_graph=False):
    return gradient_tracking_3(W, X0, e, alpha, theta, mu, calc_delta_Fx, max_iter=max_iter, use_logging=use_logging, need_graph=need_graph)