import numpy as np

def print_logs(k : int, err : float, s : np.matrix, x : np.matrix):
    print('\nk:', k, '\t', err)
    print('s:\t', s.tolist())
    print('x:\t', x.tolist())

def gradient_tracking(F : list, f : list, calc_delta_Fx, calc_error, 
    W : np.matrix, x0 : np.matrix, alpha : float, theta : float, mu : float,
    err : float, max_iter : int = 100, need_log : bool = False) -> np.matrix:

    y0 = x0.copy()
    z0 = x0.copy()
    s0 = calc_delta_Fx(f, x0)
    x = W * z0 + (1-theta) * W * x0

    y = theta * z0 + (1-theta)*x
    s = s0
    z = W * z0 - (alpha / (theta + mu * alpha)) * s
    x0 = x.copy()
    x = W * z + (1-theta) * W * x0
    if need_log : print_logs(0, calc_error(F, x, x0), s, x)

    k = 0 
    while calc_error(F, x, x0) > err and k < max_iter:
        k += 1
        y0 = y.copy()
        y = theta * z + (1-theta)*x
        s = W * s + calc_delta_Fx(f, y) - calc_delta_Fx(f, y0)
        x0 = x.copy()
        z = (1/(1+mu*alpha/theta)) * ( W * ((mu*alpha/theta) * y + z) - alpha / theta * s )
        x = W * z + (1-theta) * W * x
        if need_log : print_logs(k, calc_error(F, x, x0), s, x)
    return x