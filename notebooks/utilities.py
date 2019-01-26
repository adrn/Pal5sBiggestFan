import numpy as np
def get_uniform_idx(x, y, spacing, max_iter=10000):
    i = 0
    uniform_idx = [i]
    for k in range(max_iter):
        j = 1
        tot_dist = 0
        while tot_dist < spacing:
            idx = np.arange(i, j+1, 1, dtype=int)
            tot_dist = np.sqrt((x[idx[1:]] - x[idx[:-1]])**2 + 
                               (y[idx[1:]] - y[idx[:-1]])**2).sum()

            if j >= (len(x) - 1):
                break

            j += 1

        i = j
        if i >= (len(x) - 1):
            break
        uniform_idx.append(i)

    return np.array(uniform_idx)