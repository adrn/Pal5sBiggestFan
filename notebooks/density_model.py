import numpy as np
from scipy.special import logsumexp


def ln_normal_2d(x, mu, cov):
    x = np.array(x)
    mu = np.array(mu)
    _, logdet = np.linalg.slogdet(cov)
    Cinv = np.linalg.inv(cov)
    quad_form = (x - mu).T.dot(Cinv).dot(x - mu)
    return -0.5 * (logdet + quad_form + 2 * np.log(2*np.pi))


def get_cov(h, w):
    return np.diag([h**2, w**2])


def get_R(i, i0, N, x, y):
    j = i + i0

    dxy = np.empty(2)
    if i == 0:
        dxy[0] = x[j+1] - x[j]
        dxy[1] = y[j+1] - y[j]
    elif i == (N-1):
        dxy[0] = x[j] - x[j-1]
        dxy[1] = y[j] - y[j-1]
    else:
        dxy[0] = x[j+1] - x[j-1]
        dxy[1] = y[j+1] - y[j-1]
    dxy /= np.linalg.norm(dxy)
    theta = np.arctan2(dxy[1], dxy[0])

    R = np.empty((2, 2))
    R[0, 0] = R[1, 1] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)

    return R


def evaluate_stream_model(fit, data, x, y):
    phi1 = data['phi1_nodes']
    phi2 = data['phi2_nodes_init'] + fit['d_phi2_nodes']
    log_amp = fit['log_a_nodes']
    w = np.exp(fit['log_w_nodes'])
    h = data['h_nodes']

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    ln_val = np.zeros((data['t_n_nodes'] + data['l_n_nodes'], len(x)))
    for i in range(data['t_n_nodes']):
        C = get_cov(h[i], w[i])
        R = get_R(i, 0, data['t_n_nodes'], phi1, phi2)
        C = R @ C @ R.T
        ln_val[i] = [log_amp[i] + ln_normal_2d([x[n], y[n]],
                                               mu=[phi1[i], phi2[i]], cov=C)
                     for n in range(len(x))]

    i0 = data['t_n_nodes']
    for i in range(data['l_n_nodes']):
        j = i + i0
        C = get_cov(h[j], w[j])
        R = get_R(i, i0, data['l_n_nodes'], phi1, phi2)
        C = R @ C @ R.T

        ln_val[j] = [log_amp[j] + ln_normal_2d([x[n], y[n]],
                                               mu=[phi1[j], phi2[j]], cov=C)
                     for n in range(len(x))]

    return logsumexp(ln_val, axis=0)


def evaluate_bg_model(fit, data, x, y):
    phi1 = data['bg_phi1_nodes']
    phi2 = data['bg_phi2_nodes']
    log_amp = fit['bg_log_a_nodes']

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    C = get_cov(data['bg_h'], data['bg_w'])

    ln_val = np.zeros((data['n_bg_nodes'], len(x)))
    for i in range(data['n_bg_nodes']):
        ln_val[i] = [log_amp[i] + ln_normal_2d([x[n], y[n]],
                                               mu=[phi1[i], phi2[i]], cov=C)
                     for n in range(len(x))]

    return logsumexp(ln_val, axis=0)