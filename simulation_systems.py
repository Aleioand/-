import numpy as np
from helpers import pink_noise


def check_if_unstable(xM):
    '''
    check if generated system is stable
    '''
    try:
        test1 = np.any(abs(xM) > 1e4)
        test2 = np.any(np.isnan(xM))
        return test1 or test2
    except Exception as exc:
        print(exc)
        return True, True

def coupledKlogisticmaps(n, cM, a=4, sigma=0):
    k, tmp = cM.shape
    assert cM.shape[0] == cM.shape[1], "cM must be square!"
    ntrans = 100
    xM = np.full(shape=(n + ntrans, k), fill_value=np.nan)
    xM[0, :] = np.random.normal(0, 1, size=(1, k))
    for t in np.arange(1, n+ntrans):
        for i in np.arange(k):
            jV = np.setdiff1d(np.arange(k), i)
            tmp = xM[t - 1, i] * (a - a * xM[t - 1, i] - np.dot(cM[jV, i].T, xM[t - 1, jV].T) - sigma * np.random.normal())
            xM[t, i] = np.mod(tmp, 1)
    xM = xM[ntrans:n + ntrans, ]

    adjmatrix = cM.copy()
    adjmatrix[adjmatrix > 0] = 1
    if check_if_unstable(xM):
        return None, None
    return xM, adjmatrix


def coupledKlogisticmaps_reflect(n, cM, a=4, sigma=0):
    k, tmp = cM.shape
    assert cM.shape[0] == cM.shape[1], "cM must be square!"
    ntrans = 100
    xM = np.full(shape=(n + ntrans, k), fill_value=np.nan)
    xM[0, :] = np.random.normal(0, 1, size=(1, k))
    for t in np.arange(1, n+ntrans):
        for i in np.arange(k):
            jV = np.setdiff1d(np.arange(k), i)
            tmp = xM[t - 1, i] * (a - a * xM[t - 1, i] - np.dot(cM[jV, i].T, xM[t - 1, jV].T) - sigma * np.random.normal())
            if tmp < 0:
                xM[t, i] = np.abs(tmp)
            else:
                xM[t, i] = np.mod(tmp, 1)
    xM = xM[ntrans:n + ntrans, ]

    adjmatrix = cM.copy()
    adjmatrix[adjmatrix > 0] = 1
    if check_if_unstable(xM):
        return None, None
    return xM, adjmatrix



def coupledKlogisticsq_maps(n, cM, a=4, sigma=0):
    k, tmp = cM.shape
    assert cM.shape[0] == cM.shape[1], "cM must be square!"
    ntrans = 100
    xM = np.full(shape=(n + ntrans, k), fill_value=np.nan)
    xM[0, :] = np.random.normal(0, 1, size=(1, k))
    for t in np.arange(1, n+ntrans):
        for i in np.arange(k):
            jV = np.setdiff1d(np.arange(k), i)
            tmp = xM[t - 1, i] * (a - a * xM[t - 1, i] - np.dot(cM[jV, i].T, (xM[t - 1, jV].T) ** 2) - sigma * np.random.normal())
            xM[t, i] = np.mod(tmp, 1)
    xM = xM[ntrans:n + ntrans, ]

    adjmatrix = cM.copy()
    adjmatrix[adjmatrix > 0] = 1
    if check_if_unstable(xM):
        return None, None
    return xM, adjmatrix

######VAR systems#######
def winterhalder(n=2048):
    ntrans = 100
    K = 4
    P = 5
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.8 * xM[t - 1, 0] + 0.65 * xM[t - 4, 1] + wM[t, 0]
        xM[t, 1] = 0.6 * xM[t - 1, 1] + 0.6 * xM[t - 5, 3] + wM[t, 1]
        xM[t, 2] = 0.5 * xM[t - 3, 2] - 0.6 * xM[t - 1, 0] + 0.4 * xM[t - 4, 1] + wM[t, 2]
        xM[t, 3] = 1.2 * xM[t - 1, 3] - 0.7 * xM[t - 2, 3] + wM[t, 3]

    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[1, 0, 1, 0],
                          [1, 1, 1, 0],
                          [0, 0, 1, 0],
                          [0, 1, 0, 1]]
                         )
    if check_if_unstable(xM):
        return None, None
    return xM, adjmatrix


def coupled_randomized_signals(n=2048, c=0.5):
    '''
    Heyse et al.
    x1,t = w1,t
    x2,t = (1 − c)w2,t + c · x1,t−3
    x3,t = (1 − c)w3,t + c · x1,t−2
    x4,t = w4,t
    x5,t = (1 − c)w5,t + c · x4,t−5
    '''
    # drivers_strength_std = []
    # for _ in range(100):
    #     for n in [512, 1024, 2048]:
    ntrans = 100
    K = 5
    P = 5
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = wM[t, 0]
        xM[t, 1] = c * xM[t - 3, 0] + (1 - c) * wM[t, 1]
        xM[t, 2] = c * xM[t - 2, 0] + (1 - c) * wM[t, 2]
        xM[t, 3] = wM[t, 3]
        xM[t, 4] = c * xM[t - 5, 3] + (1 - c) * wM[t, 4]

    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0]]
                         )
    if check_if_unstable(xM):
        return None, None
    # # # #x1
    # y1 = xM[3:, [1]]
    # y1_x0 = c * xM[:-3, [0]]
    # mi1_x0 = mi_estimator_ksg1(xV=y1, yV=y1_x0)
    # mi1_x0_std = mi1_x0[0] * np.std(y1_x0) / np.std(y1)
    # print(f'mi1_x0:{mi1_x0_std}')
    # print('#####')
    # # # #x2
    # y2 = xM[2:, [2]]
    # y2_x0 = c * xM[:-2, [0]]
    # mi2_x0 = mi_estimator_ksg1(xV=y2, yV=y2_x0)
    # mi2_x0_std = mi2_x0[0] * np.std(y2_x0) / np.std(y2)
    # print(f'mi2_x0:{mi2_x0_std}')
    # print('#####')
    # # # #x4
    # y4 = xM[5:, [4]]
    # y4_x0 = c * xM[:-5, [3]]
    # mi4_x0 = mi_estimator_ksg1(xV=y4, yV=y4_x0)
    # mi4_x0_std = mi4_x0[0] * np.std(y4_x0) / np.std(y4)
    # print(f'mi4_x0:{mi4_x0_std}')
    # print('#####')
    # result_ = dict(n=n, mi1_x0_std=mi1_x0_std, mi2_x0_std=mi2_x0_std, mi4_x0_std=mi4_x0_std)
    # drivers_strength_std.append(result_)

    return xM, adjmatrix


#####NLVAR systems

def sysRunge15(n=2048, c=0.4, b=2, a=0.4, sigma=0.5):
    ntrans = 100
    K_W = 4 # #sum variables order
    K_Z = 3 # #product (synergy) variables order
    K_H = 3 # #noise order
    K = K_W + K_H + K_Z # #system's variables order
    P = 2 # #system's lag
    wM = np.random.normal(size=(n + ntrans, K_W))
    zM = np.random.normal(size=(n + ntrans, K_Z))
    hM = np.random.normal(size=(n + ntrans, K_H))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = np.concatenate([zM[:P, :], wM[:P, :], hM[:P, :]], axis=1)

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = zM[t, 0]
        xM[t, 1] = zM[t, 1]
        xM[t, 2] = zM[t, 2]
        xM[t, 3] = wM[t, 0]
        xM[t, 4] = wM[t, 1]
        xM[t, 5] = wM[t, 2]
        xM[t, 6] = wM[t, 3]
        xM[t, 7] = c * (xM[t - 2, 3] + xM[t - 2, 4] + xM[t - 2, 5] + xM[t - 2, 6]) + b * (xM[t - 2, 0] * xM[t - 2, 1] * xM[t - 2, 2]) + sigma * hM[t, 0]
        xM[t, 8] = a * (xM[t - 1, 3] + xM[t - 1, 5]) + hM[t, 1]
        xM[t, 9] = a * (xM[t - 1, 4] + xM[t - 1, 6]) + hM[t, 2]

    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([
                            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
                         )
    if check_if_unstable(xM):
        return None, None

    return xM, adjmatrix

def sysProdOnly(n=2048):
    # mi2_total = []
    # mi2_0 = []
    # mi2_1 = []
    # for _ in range(100):

    ntrans = 100
    K = 3
    P = 2
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = wM[t, 0]
        xM[t, 1] = wM[t, 1]
        xM[t, 2] = 0.5 * xM[t - 1, 0] * xM[t - 2, 1] + wM[t, 2]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[0, 0, 1],
                          [0, 0, 1],
                          [0, 0, 0]
                          ]
                         )
    if check_if_unstable(xM):
        return None, None

        # from helpers import mi_estimator_ksg1, mi_estimator_from_projection, plot_mv_timeseries
        # mi2_all, _, _ = mi_estimator_ksg1(xV=xM[2:, [2]], yV=np.concatenate([xM[1:-1, [0]], xM[:-2, [1]]], axis=1))
        # mi2_0_ = mi_estimator_from_projection(xV=xM[2:, [2]] , yV=xM[1:-1, [0]], zM=np.concatenate([xM[1:-1, [0]], xM[:-2, [1]]], axis=1))
        # mi2_1_ = mi_estimator_from_projection(xV=xM[2:, [2]] , yV=xM[1:-1, [1]], zM=np.concatenate([xM[1:-1, [0]], xM[:-2, [1]]], axis=1))
        # mi2_total.append(mi2_all)
        # mi2_0.append(mi2_0_)
        # mi2_1.append(mi2_1_)

    return xM, adjmatrix

def nlvar3K5(n=2048):
    ntrans = 100
    K = 5
    P = 3
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.7 * xM[t-1, 0] + wM[t, 0]
        xM[t, 1] = 0.3 * (xM[t-2, 0] ** 2) + wM[t, 1]
        xM[t, 2] = 0.4 * xM[t-3, 0] - 0.3 * xM[t-2, 2] + wM[t, 2]
        xM[t, 3] = 0.7 * xM[t-1, 3] - 0.3 * xM[t-1, 4]*np.exp((-xM[t-1, 4]**2)/2) + wM[t, 3]
        xM[t, 4] = 0.5 * xM[t-1, 3] + 0.2 * xM[t-2, 4] + wM[t, 4]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 1, 0],
                          ]
                         )
    if check_if_unstable(xM):
        return None, None

    return xM, adjmatrix




def sysExclusiveSynergy(n=2048):
    ntrans = 100
    K = 3
    P = 1
    wM = np.random.uniform(low=-1, high=1, size=(n + ntrans, K))
    # wM = np.random.normal(0, 1, size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = wM[t, 0]
        xM[t, 1] = wM[t, 1]
        xM[t, 2] = np.sign(xM[t - 1, 0] * xM[t - 1, 1]) + wM[t, 2]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[0, 0, 1],
                          [0, 0, 1],
                          [0, 0, 0]
                          ]
                         )
    if check_if_unstable(xM):
        return None, None

    return xM, adjmatrix


def sysVarAdditiveVlachos(alpha=1, beta=5, n=2048):
    # mi2_total = []
    # mi2_0 = []
    # mi2_1 = []
    # for _ in range(100):

    ntrans = 100
    K = 3
    P = 2
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = wM[t, 0]
        xM[t, 1] = wM[t, 1]
        xM[t, 2] = alpha * xM[t - 1, 0] + alpha * xM[t - 2, 1] + beta * wM[t, 2]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[0, 0, 1],
                          [0, 0, 1],
                          [0, 0, 0]
                          ]
                         )
    if check_if_unstable(xM):
        return None, None

        # from helpers import mi_estimator_ksg1, mi_estimator_from_projection, plot_mv_timeseries
        # mi2_all, _, _ = mi_estimator_ksg1(xV=xM[2:, [2]], yV=np.concatenate([xM[1:-1, [0]], xM[:-2, [1]]], axis=1))
        # mi2_0_ = mi_estimator_from_projection(xV=xM[2:, [2]] , yV=xM[1:-1, [0]], zM=np.concatenate([xM[1:-1, [0]], xM[:-2, [1]]], axis=1))
        # mi2_1_ = mi_estimator_from_projection(xV=xM[2:, [2]] , yV=xM[1:-1, [1]], zM=np.concatenate([xM[1:-1, [0]], xM[:-2, [1]]], axis=1))
        # mi2_total.append(mi2_all)
        # mi2_0.append(mi2_0_)
        # mi2_1.append(mi2_1_)

    return xM, adjmatrix


def sysVarAdditive3Vlachos(n=2048):
    ntrans = 100
    K = 3
    P = 1
    wM = np.random.uniform(low=-1, high=1, size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = wM[t, 0]
        xM[t, 1] = wM[t, 1]
        xM[t, 2] = xM[t - 1, 0] + xM[t - 1, 1] + wM[t, 2]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[0, 0, 1],
                          [0, 0, 1],
                          [0, 0, 0]
                          ]
                         )
    if check_if_unstable(xM):
        return None, None

    return xM, adjmatrix


def sysVarSynergy1Vlachos(n=2048):
    ntrans = 100
    K = 3
    P = 1
    wM = np.random.uniform(low=-1, high=1, size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = wM[t, 0]
        xM[t, 1] = wM[t, 1]
        xM[t, 2] = xM[t - 1, 0] * xM[t - 1, 1] + wM[t, 2]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[0, 0, 1],
                          [0, 0, 1],
                          [0, 0, 0]
                          ]
                         )
    if check_if_unstable(xM):
        return None, None

    return xM, adjmatrix

def sysVarSynergy1Vlachos3comp(n=2048):
    ntrans = 100
    K = 4
    P = 1
    # wM = np.random.uniform(low=-1, high=1, size=(n + ntrans, K))
    wM = np.random.normal(0, 1, size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = wM[t, 0]
        xM[t, 1] = wM[t, 1]
        xM[t, 2] = wM[t, 2]
        xM[t, 3] = xM[t - 1, 0] * xM[t - 1, 1] * xM[t - 1, 2] + wM[t, 3]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[0, 0, 0, 1],
                          [0, 0, 0, 1],
                          [0, 0, 0, 1],
                          [0, 0, 0, 0]
                          ]
                         )
    if check_if_unstable(xM):
        return None, None

    return xM, adjmatrix


def sysProdExp(n=2048):
    '''
    Gourevitch et al. 2006, Model 7
    from Papana2016
    '''
    ntrans = 100
    K = 3
    P = 1
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 3.4 * xM[t - 1, 0] * ((1 - xM[t - 1, 0])**2) * np.exp(-xM[t - 1, 0]**2) + 0.4 * wM[t, 0]
        xM[t, 1] = 3.4 * xM[t - 1, 1] * ((1 - xM[t - 1, 1])**2) * np.exp(-xM[t - 1, 1]**2) + 0.5 * xM[t - 1, 0] * xM[t - 1, 1] + 0.4 * wM[t, 1]
        xM[t, 2] = 3.4 * xM[t - 1, 2] * ((1 - xM[t - 1, 2])**2) * np.exp(-xM[t - 1, 2]**2) + 0.3 * xM[t - 1, 1] + 0.5 * xM[t - 1, 0] ** 2 + 0.4 * wM[t, 2]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[0, 1, 1],
                          [0, 0, 1],
                          [0, 0, 0]
                          ]
                         )
    if check_if_unstable(xM):
        return None, None

    return xM, adjmatrix


def sysProdOnly_v2(n=2048):
    '''
    Gourevitch et al. 2006, Model 7
    from Papana2016
    '''
    ntrans = 100
    K = 3
    P = 2
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.5 * xM[t - 1, 0] + wM[t, 0]
        xM[t, 1] = 0.5 * xM[t - 2, 1] + wM[t, 1]
        xM[t, 2] = 0.4 * (xM[t - 2, 0] ** 3) * xM[t - 2, 1] ** 2 + wM[t, 2]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[0, 0, 1],
                          [0, 0, 1],
                          [0, 0, 0]
                          ]
                         )

    if check_if_unstable(xM):
        return None, None

    return xM, adjmatrix


def sysProdVarm1_v5(n=2048):

    ntrans = 100
    K = 4
    P = 4
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.5 * xM[t - 1, 0] + 0.2 * (xM[t - 3, 1]**2) * xM[t - 2, 0] + wM[t, 0]
        xM[t, 1] = 0.5 * xM[t - 1, 1] + 0.2 * xM[t - 2, 2] * xM[t - 3, 3] + wM[t, 1]
        xM[t, 2] = 0.5 * xM[t - 1, 2] + 0.2 * xM[t - 2, 0] + 0.23 * xM[t - 4, 2] + wM[t, 2]
        xM[t, 3] = 0.5 * xM[t - 1, 3] + 0.2 * xM[t - 2, 2] + wM[t, 3]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[1, 0, 1, 0],
                          [1, 1, 0, 0],
                          [0, 1, 1, 1],
                          [0, 1, 0, 1]]
                         )
    if check_if_unstable(xM):
        return None, None

    return xM, adjmatrix


def sysProdVarm1_v6(n=2048):

    ntrans = 100
    K = 4
    P = 4
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.5 * xM[t - 1, 0] + 0.2 * np.exp(-xM[t - 3, 1]) * xM[t - 2, 0] + wM[t, 0]
        xM[t, 1] = 0.5 * xM[t - 1, 1] + 0.2 * np.exp(-xM[t - 2, 2] * xM[t - 3, 3]) + wM[t, 1]
        xM[t, 2] = 0.5 * xM[t - 1, 2] + 0.2 * xM[t - 2, 0] + 0.23 * xM[t - 4, 2] + wM[t, 2]
        xM[t, 3] = 0.5 * xM[t - 1, 3] + 0.2 * xM[t - 2, 2] + wM[t, 3]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[1, 0, 1, 0],
                          [1, 1, 0, 0],
                          [0, 1, 1, 1],
                          [0, 1, 0, 1]]
                         )
    if check_if_unstable(xM):
        return None, None

    return xM, adjmatrix


def sysProdVarm1(n=2048):
    ntrans = 100
    K = 4
    P = 4
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.2003 * xM[t - 1, 0] + 0.2003 * xM[t - 3, 1] * xM[t - 2, 0] + wM[t, 0]
        xM[t, 1] = 0.2003 * xM[t - 1, 1] + 0.2003 * xM[t - 2, 2] * xM[t - 3, 3] + wM[t, 1]
        xM[t, 2] = 0.2003 * xM[t - 1, 2] + 0.2003 * xM[t - 2, 0] + 0.2003 * xM[t - 4, 2] + wM[t, 2]
        xM[t, 3] = 0.2003 * xM[t - 1, 3] + 0.2003 * xM[t - 2, 2] + wM[t, 3]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[1, 0, 1, 0],
                          [1, 1, 0, 0],
                          [0, 1, 1, 1],
                          [0, 1, 0, 1]]
                         )
    if check_if_unstable(xM):
        return None, None

    # ###drivers' strength analysis
    # drivers_strength_std = []
    # # #x0
    # y0 = xM[3:, [0]]
    # y0_x0 = 0.4 * xM[:-2, [0]]
    # y0_x1 = 0.2 * xM[:-2, [1]] ** 2
    # mi0_x0 = mi_estimator_ksg1(xV=y0, yV=y0_x0)
    # print(f'mi0_x0:{mi0_x0[0] * np.std(y0_x0) / np.std(y0)}')
    # mi0_x1 = mi_estimator_ksg1(xV=y0, yV=y0_x1)
    # print(f'mi0_x1:{mi0_x1[0] * np.std(y0_x1) / np.std(y0)}')
    # print('#####')
    # # #x1
    # y1 = xM[2:, [1]]
    # y1_x0 = 0.4 * xM[:-2, [1]]
    # y1_x1 = 0.25 * xM[1:-1, [2]]
    # y1_x2 = 0.2 * xM[:-2, [3]] ** 2
    # mi1_x0 = mi_estimator_ksg1(xV=y1, yV=y1_x0)
    # print(f'mi1_x0:{mi1_x0[0] * np.std(y1_x0)/np.std(y1)}')
    # mi1_x1 = mi_estimator_ksg1(xV=y1, yV=y1_x1)
    # print(f'mi1_x1:{mi1_x1[0] * np.std(y1_x1)/np.std(y1)}')
    # mi1_x2 = mi_estimator_ksg1(xV=y1, yV=y1_x2)
    # print(f'mi1_x2:{mi1_x2[0] * np.std(y1_x2)/np.std(y1)}')
    # print('#####')
    # # #x2
    # y2 = xM[3:, [2]]
    # y2_x0 = 0.4 * xM[2:-1, [2]]
    # y2_x1 = 0.2 * xM[:-3, [1]]
    # mi2_x0 = mi_estimator_ksg1(xV=y2, yV=y2_x0)
    # print(f'mi2_x0:{mi2_x0[0] * np.std(y2_x0) / np.std(y2)}')
    # mi2_x1 = mi_estimator_ksg1(xV=y2, yV=y2_x1)
    # print(f'mi2_x1:{mi2_x1[0] * np.std(y2_x1) / np.std(y2)}')
    # print('#####')
    # # #x3
    # y3 = xM[3:, [3]]
    # y3_x0 = 0.4 * xM[1:-2, [3]]
    # y3_x1 = 0.21 * xM[:-3, [0]]
    # mi3_x0 = mi_estimator_ksg1(xV=y3, yV=y3_x0)
    # print(f'mi3_x0:{mi3_x0[0] * np.std(y3_x0) / np.std(y3)}')
    # mi3_x1 = mi_estimator_ksg1(xV=y3, yV=y3_x1)
    # print(f'mi3_x1:{mi3_x1[0] * np.std(y3_x1) / np.std(y3)}')

    return xM, adjmatrix


def sysSquaredVarm1(n=2048):
    ntrans = 100
    K = 4
    P = 3
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.2108 * xM[t - 2, 0] + 0.2108 * xM[t - 2, 1] * xM[t - 2, 1] + wM[t, 0]
        xM[t, 1] = 0.2108 * xM[t - 2, 1] + 0.2108 * xM[t - 1, 2] + 0.2108 * xM[t - 2, 3] * xM[t - 2, 3] + wM[t, 1]
        xM[t, 2] = 0.2108 * xM[t - 2, 2] + 0.2108 * xM[t - 3, 1] + wM[t, 2]
        xM[t, 3] = 0.2108 * xM[t - 2, 3] + 0.2108 * xM[t - 3, 0] + wM[t, 3]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[1, 0, 0, 1],
                          [1, 1, 1, 0],
                          [0, 1, 1, 0],
                          [0, 1, 0, 1]]
                         )
    if check_if_unstable(xM):
        return None, None

    # ###drivers' strength analysis
    # drivers_strength_std = []
    # # #x0
    # y0 = xM[2:, [0]]
    # y0_x0 = 0.4 * xM[:-2, [0]]
    # y0_x1 = 0.2 * xM[:-2, [1]] ** 2
    # mi0_x0 = mi_estimator_ksg1(xV=y0, yV=y0_x0)
    # print(f'mi0_x0:{mi0_x0[0] * np.std(y0_x0) / np.std(y0)}')
    # mi0_x1 = mi_estimator_ksg1(xV=y0, yV=y0_x1)
    # print(f'mi0_x1:{mi0_x1[0] * np.std(y0_x1) / np.std(y0)}')
    # print('#####')
    # # #x1
    # y1 = xM[2:, [1]]
    # y1_x0 = 0.4 * xM[:-2, [1]]
    # y1_x1 = 0.25 * xM[1:-1, [2]]
    # y1_x2 = 0.2 * xM[:-2, [3]] ** 2
    # mi1_x0 = mi_estimator_ksg1(xV=y1, yV=y1_x0)
    # print(f'mi1_x0:{mi1_x0[0] * np.std(y1_x0)/np.std(y1)}')
    # mi1_x1 = mi_estimator_ksg1(xV=y1, yV=y1_x1)
    # print(f'mi1_x1:{mi1_x1[0] * np.std(y1_x1)/np.std(y1)}')
    # mi1_x2 = mi_estimator_ksg1(xV=y1, yV=y1_x2)
    # print(f'mi1_x2:{mi1_x2[0] * np.std(y1_x2)/np.std(y1)}')
    # print('#####')
    # # #x2
    # y2 = xM[3:, [2]]
    # y2_x0 = 0.4 * xM[2:-1, [2]]
    # y2_x1 = 0.2 * xM[:-3, [1]]
    # mi2_x0 = mi_estimator_ksg1(xV=y2, yV=y2_x0)
    # print(f'mi2_x0:{mi2_x0[0] * np.std(y2_x0) / np.std(y2)}')
    # mi2_x1 = mi_estimator_ksg1(xV=y2, yV=y2_x1)
    # print(f'mi2_x1:{mi2_x1[0] * np.std(y2_x1) / np.std(y2)}')
    # print('#####')
    # # #x3
    # y3 = xM[3:, [3]]
    # y3_x0 = 0.4 * xM[1:-2, [3]]
    # y3_x1 = 0.21 * xM[:-3, [0]]
    # mi3_x0 = mi_estimator_ksg1(xV=y3, yV=y3_x0)
    # print(f'mi3_x0:{mi3_x0[0] * np.std(y3_x0) / np.std(y3)}')
    # mi3_x1 = mi_estimator_ksg1(xV=y3, yV=y3_x1)
    # print(f'mi3_x1:{mi3_x1[0] * np.std(y3_x1) / np.std(y3)}')

    return xM, adjmatrix


def sysProdVarm1_v4(n=2048):
    ntrans = 100
    K = 4
    P = 4
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.5 * xM[t - 1, 0] + 0.2 * xM[t - 3, 1] * xM[t - 2, 0] + wM[t, 0]
        xM[t, 1] = 0.5 * xM[t - 1, 1] + 0.2 * xM[t - 2, 2] * xM[t - 3, 3] + wM[t, 1]
        xM[t, 2] = 0.5 * xM[t - 1, 2] + 0.2 * xM[t - 2, 0] + 0.23 * xM[t - 4, 2] + wM[t, 2]
        xM[t, 3] = 0.5 * xM[t - 1, 3] + 0.2 * xM[t - 2, 2] + wM[t, 3]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[1, 0, 1, 0],
                          [1, 1, 0, 0],
                          [0, 1, 1, 1],
                          [0, 1, 0, 1]]
                         )
    if check_if_unstable(xM):
        return None, None

    # ###drivers' strength analysis
    # drivers_strength_std = []
    # # #x0
    # y0 = xM[3:, [0]]
    # y0_x0 = 0.4 * xM[:-2, [0]]
    # y0_x1 = 0.2 * xM[:-2, [1]] ** 2
    # mi0_x0 = mi_estimator_ksg1(xV=y0, yV=y0_x0)
    # print(f'mi0_x0:{mi0_x0[0] * np.std(y0_x0) / np.std(y0)}')
    # mi0_x1 = mi_estimator_ksg1(xV=y0, yV=y0_x1)
    # print(f'mi0_x1:{mi0_x1[0] * np.std(y0_x1) / np.std(y0)}')
    # print('#####')
    # # #x1
    # y1 = xM[2:, [1]]
    # y1_x0 = 0.4 * xM[:-2, [1]]
    # y1_x1 = 0.25 * xM[1:-1, [2]]
    # y1_x2 = 0.2 * xM[:-2, [3]] ** 2
    # mi1_x0 = mi_estimator_ksg1(xV=y1, yV=y1_x0)
    # print(f'mi1_x0:{mi1_x0[0] * np.std(y1_x0)/np.std(y1)}')
    # mi1_x1 = mi_estimator_ksg1(xV=y1, yV=y1_x1)
    # print(f'mi1_x1:{mi1_x1[0] * np.std(y1_x1)/np.std(y1)}')
    # mi1_x2 = mi_estimator_ksg1(xV=y1, yV=y1_x2)
    # print(f'mi1_x2:{mi1_x2[0] * np.std(y1_x2)/np.std(y1)}')
    # print('#####')
    # # #x2
    # y2 = xM[3:, [2]]
    # y2_x0 = 0.4 * xM[2:-1, [2]]
    # y2_x1 = 0.2 * xM[:-3, [1]]
    # mi2_x0 = mi_estimator_ksg1(xV=y2, yV=y2_x0)
    # print(f'mi2_x0:{mi2_x0[0] * np.std(y2_x0) / np.std(y2)}')
    # mi2_x1 = mi_estimator_ksg1(xV=y2, yV=y2_x1)
    # print(f'mi2_x1:{mi2_x1[0] * np.std(y2_x1) / np.std(y2)}')
    # print('#####')
    # # #x3
    # y3 = xM[3:, [3]]
    # y3_x0 = 0.4 * xM[1:-2, [3]]
    # y3_x1 = 0.21 * xM[:-3, [0]]
    # mi3_x0 = mi_estimator_ksg1(xV=y3, yV=y3_x0)
    # print(f'mi3_x0:{mi3_x0[0] * np.std(y3_x0) / np.std(y3)}')
    # mi3_x1 = mi_estimator_ksg1(xV=y3, yV=y3_x1)
    # print(f'mi3_x1:{mi3_x1[0] * np.std(y3_x1) / np.std(y3)}')

    return xM, adjmatrix


def sysSquaredVarm1_v2(n=2048):
    ntrans = 100
    K = 4
    P = 3
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.4 * xM[t - 2, 0] + 0.2 * xM[t - 2, 1] * xM[t - 2, 1] + wM[t, 0]
        xM[t, 1] = 0.4 * xM[t - 2, 1] + 0.25 * xM[t - 1, 2] + 0.2 * xM[t - 2, 3] * xM[t - 2, 3] + wM[t, 1]
        xM[t, 2] = 0.4 * xM[t - 2, 2] + 0.2 * xM[t - 3, 1] + wM[t, 2]
        xM[t, 3] = 0.4 * xM[t - 2, 3] + 0.21 * xM[t - 3, 0] + wM[t, 3]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[1, 0, 0, 1],
                          [1, 1, 1, 0],
                          [0, 1, 1, 0],
                          [0, 1, 0, 1]]
                         )
    if check_if_unstable(xM):
        return None, None

    # ###drivers' strength analysis
    # drivers_strength_std = []
    # # #x0
    # y0 = xM[2:, [0]]
    # y0_x0 = 0.4 * xM[:-2, [0]]
    # y0_x1 = 0.2 * xM[:-2, [1]] ** 2
    # mi0_x0 = mi_estimator_ksg1(xV=y0, yV=y0_x0)
    # print(f'mi0_x0:{mi0_x0[0] * np.std(y0_x0) / np.std(y0)}')
    # mi0_x1 = mi_estimator_ksg1(xV=y0, yV=y0_x1)
    # print(f'mi0_x1:{mi0_x1[0] * np.std(y0_x1) / np.std(y0)}')
    # print('#####')
    # # #x1
    # y1 = xM[2:, [1]]
    # y1_x0 = 0.4 * xM[:-2, [1]]
    # y1_x1 = 0.25 * xM[1:-1, [2]]
    # y1_x2 = 0.2 * xM[:-2, [3]] ** 2
    # mi1_x0 = mi_estimator_ksg1(xV=y1, yV=y1_x0)
    # print(f'mi1_x0:{mi1_x0[0] * np.std(y1_x0)/np.std(y1)}')
    # mi1_x1 = mi_estimator_ksg1(xV=y1, yV=y1_x1)
    # print(f'mi1_x1:{mi1_x1[0] * np.std(y1_x1)/np.std(y1)}')
    # mi1_x2 = mi_estimator_ksg1(xV=y1, yV=y1_x2)
    # print(f'mi1_x2:{mi1_x2[0] * np.std(y1_x2)/np.std(y1)}')
    # print('#####')
    # # #x2
    # y2 = xM[3:, [2]]
    # y2_x0 = 0.4 * xM[2:-1, [2]]
    # y2_x1 = 0.2 * xM[:-3, [1]]
    # mi2_x0 = mi_estimator_ksg1(xV=y2, yV=y2_x0)
    # print(f'mi2_x0:{mi2_x0[0] * np.std(y2_x0) / np.std(y2)}')
    # mi2_x1 = mi_estimator_ksg1(xV=y2, yV=y2_x1)
    # print(f'mi2_x1:{mi2_x1[0] * np.std(y2_x1) / np.std(y2)}')
    # print('#####')
    # # #x3
    # y3 = xM[3:, [3]]
    # y3_x0 = 0.4 * xM[1:-2, [3]]
    # y3_x1 = 0.21 * xM[:-3, [0]]
    # mi3_x0 = mi_estimator_ksg1(xV=y3, yV=y3_x0)
    # print(f'mi3_x0:{mi3_x0[0] * np.std(y3_x0) / np.std(y3)}')
    # mi3_x1 = mi_estimator_ksg1(xV=y3, yV=y3_x1)
    # print(f'mi3_x1:{mi3_x1[0] * np.std(y3_x1) / np.std(y3)}')

    return xM, adjmatrix


def sysSquaredVarm1_v4(n=2048):
    ntrans = 100
    K = 4
    P = 3
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.4 * xM[t - 2, 0] + 0.3 * xM[t - 2, 1] * xM[t - 2, 1] + wM[t, 0]
        xM[t, 1] = 0.4 * xM[t - 2, 1] + 0.25 * xM[t - 1, 2] + 0.3 * xM[t - 2, 3] * xM[t - 2, 3] + wM[t, 1]
        xM[t, 2] = 0.4 * xM[t - 2, 2] + 0.2 * xM[t - 3, 1] + wM[t, 2]
        xM[t, 3] = 0.4 * xM[t - 2, 3] + 0.2 * xM[t - 3, 0] + wM[t, 3]
    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[1, 0, 0, 1],
                          [1, 1, 1, 0],
                          [0, 1, 1, 0],
                          [0, 1, 0, 1]]
                         )
    if check_if_unstable(xM):
        return None, None
    return xM, adjmatrix


def sysBigMixedVarm1(n=2048):
    ntrans = 100
    K = 20
    P = 3
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.8 * xM[t - 1, 0] - 0.1 * xM[t - 2, 2] - 0.3 * xM[t - 2, 18] + wM[t, 0]
        xM[t, 1] = 0.8 * xM[t - 1, 1] - 0.2 * xM[t - 2, 0] * xM[t - 1, 14] + wM[t, 1]
        xM[t, 2] = 0.8 * xM[t - 1, 2] + wM[t, 2]
        xM[t, 3] = 0.8 * xM[t - 1, 3] - 0.2 * xM[t - 2, 4] * xM[t - 1, 6] + wM[t, 3]
        xM[t, 4] = 0.8 * xM[t - 1, 4] - 0.15 * xM[t - 1, 2] * xM[t - 1, 2] + wM[t, 4]
        xM[t, 5] = 0.8 * xM[t - 1, 5] - 0.4 * xM[t - 3, 2] + 0.1 * xM[t - 1, 9] * xM[t - 1, 9] + wM[t, 5]
        xM[t, 6] = 0.8 * xM[t - 1, 6] + wM[t, 6]
        xM[t, 7] = -0.6 * xM[t - 1, 7] + 0.1 * xM[t - 3, 10] * xM[t - 2, 18] + wM[t, 7]
        xM[t, 8] = 0.8 * xM[t - 1, 8] - 0.1 * xM[t - 1, 19] + wM[t, 8]
        xM[t, 9] = 0.8 * xM[t - 1, 9] + wM[t, 9]
        xM[t, 10] = 0.8 * xM[t - 1, 10] + 0.1 * xM[t - 1, 19] - 0.1 * xM[t - 2, 3] * xM[t - 2, 3] + wM[t, 10]
        xM[t, 11] = 0.8 * xM[t - 1, 11] + 0.2 * xM[t - 2, 18] - 0.1 * xM[t - 1, 15] * xM[t - 1, 15] + wM[t, 11]
        xM[t, 12] = 0.8 * xM[t - 1, 12] + wM[t, 12]
        xM[t, 13] = 0.8 * xM[t - 1, 13] - 0.2 * xM[t - 3, 3] * xM[t - 3, 3] + wM[t, 13]
        xM[t, 14] = 0.8 * xM[t - 1, 14] + wM[t, 14]
        xM[t, 15] = 0.8 * xM[t - 1, 15] + 0.1 * xM[t - 2, 18] - 0.3 * xM[t - 2, 2] + wM[t, 15]
        xM[t, 16] = 0.8 * xM[t - 1, 16] - 0.1 * xM[t - 2, 6] * xM[t - 2, 6] + wM[t, 16]
        xM[t, 17] = 0.8 * xM[t - 1, 17] - 0.3 * xM[t - 2, 6] + 0.5 * xM[t - 3, 1] * xM[t - 1, 19] + wM[t, 17]
        xM[t, 18] = 0.8 * xM[t - 1, 18] + wM[t, 18]
        xM[t, 19] = 0.8 * xM[t - 1, 19] + wM[t, 19]

    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                          [1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])

    if check_if_unstable(xM):
        return None, None
    return xM, adjmatrix


def sysMixedVarm1(n=2048):
    ntrans = 100
    K = 3
    P = 3
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]
    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.2336 * xM[t - 1, 0] + 0.2336 * xM[t - 3, 0] * xM[t - 2, 1] - 0.2336 * xM[t - 3, 2] * xM[t - 3, 2] + \
                   wM[t, 0]
        xM[t, 1] = 0.2336 * xM[t - 1, 1] + wM[t, 1]
        xM[t, 2] = 0.2336 * xM[t - 1, 2] - 0.2336 * xM[t - 2, 0] * xM[t - 1, 1] + wM[t, 2]
    xM = xM[ntrans + 1:, :]
    if check_if_unstable(xM):
        return None, None
    adjmatrix = np.array([
        [1., 0., 1.],
        [1., 1., 1.],
        [1., 0., 1.]]
    )
    return xM, adjmatrix


def sysK10P4(n=2048):
    ntrans = 100
    K = 10
    P = 4

    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.3 * xM[t - 1, 0] + 0.95 * xM[t - 2, 4] + wM[t, 0]
        xM[t, 1] = 0.3 * xM[t - 1, 1] + wM[t, 1]
        xM[t, 2] = 0.3 * xM[t - 1, 2] + wM[t, 2]
        xM[t, 3] = 0.3 * xM[t - 1, 3] + 0.95 * xM[t - 2, 2] + 0.95 * xM[t - 3, 1] + wM[t, 3]
        xM[t, 4] = 0.3 * xM[t - 1, 4] * xM[t - 1, 3] + 0.95 * xM[t - 4, 7] * xM[t - 4, 8] + wM[t, 4]
        xM[t, 5] = 0.95 * xM[t - 1, 2] + wM[t, 5]
        xM[t, 6] = 0.3 * xM[t - 1, 6] + wM[t, 6]
        xM[t, 7] = 0.3 * xM[t - 1, 7] + wM[t, 7]
        xM[t, 8] = 0.3 * xM[t - 1, 8] * xM[t - 1, 3] + wM[t, 8]
        xM[t, 9] = 0.95 * xM[t - 1, 0] * xM[t - 1, 2] + wM[t, 9]

    xM = xM[ntrans + 1:, :]
    if check_if_unstable(xM):
        return None, None
    adjmatrix = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 1., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
    )
    return xM, adjmatrix


def sysK10P4v1(n=2048):
    ntrans = 100
    K = 10
    P = 4

    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.3 * xM[t - 1, 0] + 0.95 * xM[t - 2, 4] + wM[t, 0]
        xM[t, 1] = 0.3 * xM[t - 1, 1] + wM[t, 1]
        xM[t, 2] = 0.3 * xM[t - 1, 2] + wM[t, 2]
        xM[t, 3] = 0.3 * xM[t - 1, 3] + 0.95 * xM[t - 2, 2] + 0.95 * xM[t - 3, 1] + wM[t, 3]
        xM[t, 4] = 0.3 * xM[t - 1, 4] * xM[t - 1, 3] + 0.3 * xM[t - 4, 7] * xM[t - 4, 8] + wM[t, 4]
        xM[t, 5] = 0.95 * xM[t - 1, 2] + wM[t, 5]
        xM[t, 6] = 0.3 * xM[t - 1, 6] + wM[t, 6]
        xM[t, 7] = 0.3 * xM[t - 1, 7] + wM[t, 7]
        xM[t, 8] = 0.3 * xM[t - 1, 8] * xM[t - 1, 3] + wM[t, 8]
        xM[t, 9] = 0.3 * xM[t - 1, 0] * xM[t - 1, 2] + wM[t, 9]

    xM = xM[ntrans + 1:, :]
    if check_if_unstable(xM):
        return None, None
    adjmatrix = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 1., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
    )
    return xM, adjmatrix


def sysK30P2(n=2048):
    ntrans = 100
    K = 30
    P = 2

    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.361 * xM[t - 1, 0] + wM[t, 0]
        xM[t, 1] = 0.366 * xM[t - 1, 1] + 0.384 * xM[t - 1, 22] * xM[t - 1, 13] + 0.383 * xM[t - 1, 26] + wM[t, 1]
        xM[t, 2] = 0.401 * xM[t - 1, 2] + 0.397 * xM[t - 1, 11] * xM[t - 1, 15] + 0.38 * xM[t - 1, 12] \
                   + 0.39 * xM[t - 1, 18] * xM[t - 1, 24] + 0.352 * xM[t - 1, 26] + 0.384 * xM[t - 1, 27] + 0.414 * xM[t - 2, 15] + wM[
                       t, 2]
        xM[t, 3] = 0.364 * xM[t - 1, 3] * xM[t - 1, 24] + wM[t, 3]
        xM[t, 4] = 0.414 * xM[t - 1, 4] * xM[t - 1, 28] + wM[t, 4]
        xM[t, 5] = 0.355 * xM[t - 1, 5] + wM[t, 5]
        xM[t, 6] = 0.411 * xM[t - 1, 6] * xM[t - 1, 16] + 0.357 * xM[t - 1, 9] + 0.37 * xM[t - 1, 18] + 0.415 * xM[
            t - 1, 19] + 0.413 * xM[t - 2, 8] + wM[t, 6]
        xM[t, 7] = 0.386 * xM[t - 1, 7] * xM[t - 1, 29] + 0.408 * xM[t - 1, 13] + wM[t, 7]
        xM[t, 8] = 0.385 * xM[t - 1, 8] + 0.373 * xM[t - 1, 11] + wM[t, 8]
        xM[t, 9] = 0.398 * xM[t - 1, 9] + 0.367 * xM[t - 2, 29] * xM[t - 2, 4] + wM[t, 9]
        xM[t, 10] = 0.387 * xM[t - 1, 10] + 0.35 * xM[t - 2, 5] + wM[t, 10]
        xM[t, 11] = 0.385 * xM[t - 1, 1] + 0.415 * xM[t - 1, 11] + wM[t, 11]
        xM[t, 12] = 0.4 * xM[t - 1, 12] + 0.365 * xM[t - 2, 11] + wM[t, 12]
        xM[t, 13] = 0.366 * xM[t - 1, 13] + wM[t, 13]
        xM[t, 14] = 0.411 * xM[t - 1, 14] * xM[t - 1, 19] + wM[t, 14]
        xM[t, 15] = 0.369 * xM[t - 1, 15] + wM[t, 15]
        xM[t, 16] = 0.367 * xM[t - 1, 16] + wM[t, 16]
        xM[t, 17] = 0.372 * xM[t - 1, 17] + 0.376 * xM[t - 2, 18] + wM[t, 17]
        xM[t, 18] = 0.38 * xM[t - 1, 6] * xM[t - 1, 17] + 0.389 * xM[t - 1, 18] + 0.371 * xM[t - 1, 19] + wM[t, 18]
        xM[t, 19] = 0.367 * xM[t - 1, 19] + 0.35 * xM[t - 1, 23] + 0.409 * xM[t - 1, 29] + wM[t, 19]
        xM[t, 20] = 0.383 * xM[t - 1, 15] + 0.395 * xM[t - 1, 20] + 0.391 * xM[t - 1, 27] + wM[t, 20]
        xM[t, 21] = 0.398 * xM[t - 1, 4] + 0.351 * xM[t - 1, 21] + 0.381 * xM[t - 2, 25] + wM[t, 21]
        xM[t, 22] = 0.362 * xM[t - 1, 22] * xM[t - 1, 7] + wM[t, 22]
        xM[t, 23] = 0.368 * xM[t - 1, 3] + 0.413 * xM[t - 1, 11] + 0.356 * xM[t - 1, 23] + wM[t, 23]
        xM[t, 24] = 0.374 * xM[t - 1, 24] + wM[t, 24]
        xM[t, 25] = 0.386 * xM[t - 1, 25] + 0.405 * xM[t - 1, 28] + 0.397 * xM[t - 2, 2] + wM[t, 25]
        xM[t, 26] = 0.353 * xM[t - 1, 5] * xM[t - 1, 22] + 0.386 * xM[t - 1, 26] + wM[t, 26]
        xM[t, 27] = 0.403 * xM[t - 1, 25] + 0.351 * xM[t - 1, 27] + 0.372 * xM[t - 2, 2] + wM[t, 27]
        xM[t, 28] = 0.409 * xM[t - 1, 28] + wM[t, 28]
        xM[t, 29] = 0.362 * xM[t - 1, 29] + wM[t, 29]

    xM = xM[ntrans + 1:, :]
    if check_if_unstable(xM):
        return None, None
    adjmatrix = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
         0., 0.],
        [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.]]
    )

    return xM, adjmatrix


def sysMedMixedVarm1(n=2048):
    ntrans = 100
    K = 10
    P = 4
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.3575 * xM[t - 2, 0] + wM[t, 0]
        xM[t, 1] = 0.3575 * xM[t - 2, 1] + 0.3575 * xM[t - 1, 3] * xM[t - 1, 3] - 0.3575 * xM[t - 2, 7] * xM[t - 4, 9] + \
                   wM[t, 1]
        xM[t, 2] = 0.3575 * xM[t - 2, 2] + 0.3575 * xM[t - 1, 4] * xM[t - 4, 0] + wM[t, 2]
        xM[t, 3] = 0.3575 * xM[t - 2, 3] + wM[t, 3]
        xM[t, 4] = 0.3575 * xM[t - 2, 4] + wM[t, 4]
        xM[t, 5] = 0.3575 * xM[t - 2, 5] - 0.3575 * xM[t - 2, 2] + 0.3575 * xM[t - 1, 8] * xM[t - 1, 8] + wM[t, 5]
        xM[t, 6] = 0.3575 * xM[t - 2, 6] + wM[t, 6]
        xM[t, 7] = 0.3575 * xM[t - 2, 7] + 0.1 * xM[t - 1, 6] * xM[t - 1, 6] + wM[t, 7]
        xM[t, 8] = 0.3575 * xM[t - 2, 8] + wM[t, 8]
        xM[t, 9] = 0.3575 * xM[t - 2, 9] + 0.3575 * xM[t - 1, 8] + wM[t, 9]

    xM = xM[ntrans + 1:, :]
    if check_if_unstable(xM):
        return None, None
    adjmatrix = np.array([
        [1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 1., 1.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 1.]]
    )
    return xM, adjmatrix


def sysMedMixedVarm1_v3(n=2048):
    ntrans = 100
    K = 10
    P = 4
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.49 * xM[t - 2, 0] + wM[t, 0]
        xM[t, 1] = 0.49 * xM[t - 2, 1] + 0.29 * xM[t - 1, 3] * xM[t - 1, 3] - 0.31 * xM[t - 2, 7] * xM[t - 4, 9] + wM[t, 1]
        xM[t, 2] = 0.49 * xM[t - 2, 2] + 0.29 * xM[t - 1, 4] * xM[t - 4, 0] + wM[t, 2]
        xM[t, 3] = 0.49 * xM[t - 2, 3] + wM[t, 3]
        xM[t, 4] = 0.49 * xM[t - 2, 4] + wM[t, 4]
        xM[t, 5] = 0.49 * xM[t - 2, 5] - 0.35 * xM[t - 2, 2] + 0.31 * xM[t - 1, 8] * xM[t - 1, 8] + wM[t, 5]
        xM[t, 6] = 0.49 * xM[t - 2, 6] + wM[t, 6]
        xM[t, 7] = 0.49 * xM[t - 2, 7] + 0.1 * xM[t - 1, 6] * xM[t - 1, 6] + wM[t, 7]
        xM[t, 8] = 0.49 * xM[t - 2, 8] + wM[t, 8]
        xM[t, 9] = 0.49 * xM[t - 2, 9] + 0.32 * xM[t - 1, 8] + wM[t, 9]

    xM = xM[ntrans + 1:, :]
    if check_if_unstable(xM):
        return None, None
    adjmatrix = np.array([
        [1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 1., 1.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 1.]]
    )
    return xM, adjmatrix


def sysMedMixedVarm1_v3_linearized(n=2048):
    ntrans = 100
    K = 10
    P = 4
    wM = np.random.normal(size=(n + ntrans, K))
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = wM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.49 * xM[t - 2, 0] + wM[t, 0]
        xM[t, 1] = 0.49 * xM[t - 2, 1] + 0.29 * xM[t - 1, 3] - 0.31 * (xM[t - 2, 7] + xM[t - 4, 9]) + wM[t, 1]
        xM[t, 2] = 0.49 * xM[t - 2, 2] + 0.29 * (xM[t - 1, 4] + xM[t - 4, 0]) + wM[t, 2]
        xM[t, 3] = 0.49 * xM[t - 2, 3] + wM[t, 3]
        xM[t, 4] = 0.49 * xM[t - 2, 4] + wM[t, 4]
        xM[t, 5] = 0.49 * xM[t - 2, 5] - 0.35 * xM[t - 2, 2] + 0.31 * xM[t - 1, 8] + wM[t, 5]
        xM[t, 6] = 0.49 * xM[t - 2, 6] + wM[t, 6]
        xM[t, 7] = 0.49 * xM[t - 2, 7] + 0.1 * xM[t - 1, 6] + wM[t, 7]
        xM[t, 8] = 0.49 * xM[t - 2, 8] + wM[t, 8]
        xM[t, 9] = 0.49 * xM[t - 2, 9] + 0.32 * xM[t - 1, 8] + wM[t, 9]

    xM = xM[ntrans + 1:, :]
    if check_if_unstable(xM):
        return None, None
    adjmatrix = np.array([
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 1., 0., 0., 0., 1., 0., 1.],
        [0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 1., 1.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 1.]]
    )
    return xM, adjmatrix



def pinkarnonlin(n=2048):
    '''
       Heyse et al.
    '''
    ntrans = 0
    K = 5
    P = 5
    thetaM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    for j in np.arange(K):
        thetaM[:, [j]] = pink_noise(n=n + ntrans, ind=1, offset=0)
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    xM[:P, :] = thetaM[:P, :]

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = 0.95 * np.sqrt(2) * xM[t - 1, 0] - 0.9025 * xM[t - 2, 0] + thetaM[t, 0]
        xM[t, 1] = 0.5 * (xM[t - 2, 0] ** 2) + thetaM[t, 1]
        xM[t, 2] = -0.4 * xM[t - 3, 0] + 0.25 * np.sqrt(2) * xM[t - 3, 1] + thetaM[t, 2]
        xM[t, 3] = -0.5 * (xM[t - 5, 0] ** 2) + 0.25 * np.sqrt(2) * xM[t - 1, 3] + 0.25 * np.sqrt(2) * xM[t - 1, 4] + \
                   thetaM[t, 3]
        xM[t, 4] = -0.25 * np.sqrt(2) * xM[t - 1, 3] + 0.25 * np.sqrt(2) * xM[t - 1, 4] + thetaM[t, 4]

    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ]
    )
    if check_if_unstable(xM):
        return None, None
    return xM, adjmatrix


###PERIODIC systems

def seizure_model(n=2048, start_f0=12, end_f0=8, snr=-5):
    '''
    Heyse et al.
    generates model that mimics the propagation of an epileptic seizure,
    where seizure activity is modeled as a sine wave with time-varying frequency
    :param n: length of timeseries
    :param start_f0: sine frequency that varies from min_f0 - max_f0
    :param end_f0:
    :param snr: noise amplitude is chosen such that the SNR equals −5dB
    :return: generated system
    x1,t = sin(2πft) + θ1,t
    x2,t = x1,t−2 + θ2,t
    x3,t = x1,t−4 + θ3,t
    x4,t = θ4,t
    x5,t = θ5,t
    '''

    ### SNR = 10log10(P_signal/P_noise)
    ###P_signal = amplitude**2
    ###P_signal = P_sine = E[X^2(t)] = var(X) = 0.5
    ntrans = 0
    K = 5
    P = 4
    thetaM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    for j in np.arange(K):
        thetaM[:, [j]] = pink_noise(n=n + ntrans, ind=1, offset=0)
    xM = np.full(shape=(n + ntrans, K), fill_value=np.nan)
    signal = np.full(shape=(n + ntrans, 1), fill_value=np.nan)
    xM[:P, :] = thetaM[:P, :]

    # #calculate linearly decreasing frequency
    # f = start_f0 + f_slope * t
    f_slope = (end_f0 - start_f0) / (n + ntrans)

    # #calculate signal
    for t in np.arange(n + ntrans):
        f = start_f0 + f_slope * t
        signal[t] = np.sin(2 * np.pi * f * t)
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(np.mean(thetaM ** 2, axis=0))
    noise_amplitude = np.sqrt(signal_power / (10 ** (snr / 10)) / noise_power)
    thetaM *= noise_amplitude  # amplify noise to get the defined SNR

    ###check if SNR is the defined
    # snr = 10 * np.log10(signal_power / np.mean(np.mean(thetaM ** 2, axis=0)))

    for t in np.arange(P, n + ntrans):
        xM[t, 0] = signal[t] + thetaM[t, 0]
        xM[t, 1] = xM[t - 2, 0] + thetaM[t, 1]
        xM[t, 2] = xM[t - 4, 0] + thetaM[t, 2]
        xM[t, 3] = thetaM[t, 3]
        xM[t, 4] = thetaM[t, 4]

    xM = xM[ntrans + 1:, :]
    adjmatrix = np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    )
    if check_if_unstable(xM):
        return None, None
    return xM, adjmatrix


###CHAOTIC systems
def anycoupledhenon(c_m, n=2 ** 10, a_v=np.array([1.4, 0.3]), x0_m=np.nan):
    k, tmp = c_m.shape
    if k != tmp:
        print('c_m must be squared!')
    if np.isnan(x0_m):
        n_trans = 0
        x_m = np.full(shape=(n + n_trans, k), fill_value=np.nan)
        x0lim_v = np.array([-1, 1, -0.3, 0.3]).reshape(-1, 1)
        tmp_m = np.random.uniform(0, 1, size=(2, k))
        x_m[0, :] = (x0lim_v[3] - tmp_m[0, :]) / (x0lim_v[3] - x0lim_v[2])
        x_m[1, :] = (x0lim_v[1] - tmp_m[1, :]) / (x0lim_v[1] - x0lim_v[0])
    else:
        n_trans = 2
        x_m = np.full(shape=(n + n_trans, k), fill_value=np.nan)
        x_m[:2, :] = x0_m

    for t in np.arange(2, n + n_trans):
        for k_ in np.arange(k):
            # x_m[t, k_] = 1 - x_m[t - 1,k_] * np.dot(x_m[t - 1,:],c_m[:,k_]) +  a_v[1] * x_m[t - 2, k_]
            x_m[t, k_] = a_v[0] - x_m[t - 1, k_] * np.dot(x_m[t - 1, :], c_m[:, k_]) + a_v[1] * x_m[t - 2, k_]
        if np.isnan(x_m[t, k_]):
            break

    x_m = x_m[n_trans:, :]
    if check_if_unstable(x_m):
        return None, None
    adjmatrix = c_m.copy()
    adjmatrix[adjmatrix > 0] = 1
    return x_m, adjmatrix


def causalhenonmap(k, c, n, aV=np.array([1.4, 0.3]), x0M=np.nan):
    '''
    xM = causalhenonmaps2(c,n,aV,x0V)
    CAUSALHENONMAPS2 generates m time series from identical uni-directional
    coupled Henon maps with coupling strength c. Each map causes the next and
    the last map does not cause any other map.
    INPUTS
    - m   : the number of coupled maps (time series)
    - c   : the coupling strength
    - n   : the time series length
    - aV  : the two parameters in a vector (if omitted the standard
            parameters for chaos are chosen)
    - x0M : initial conditions, a matrix of size 2 x m
    OUTPUTS
    - xM  : the n x m matrix of the generated time series
    '''

    cV = c[0] * np.ones(shape=(k, 1))
    if np.isnan(x0M):
        ntrans = 1000
        x0limV = np.array([-1, 1, - 0.3, 0.3])
        xM = np.full((n + ntrans, k), np.nan)
        tmpM = np.random.uniform(0, 1, (2, k))
        xM[0, :] = (x0limV[3] - tmpM[0, :]) / (x0limV[3] - x0limV[2])
        xM[1, :] = (x0limV[1] - tmpM[1, :]) / (x0limV[1] - x0limV[0])
    else:
        ntrans = 2
        xM = np.full((n + ntrans, k), np.nan)
        xM[:2, :] = x0M
    for t in range(2, n + ntrans):
        xM[t, 0] = aV[0] - xM[t - 1, 0] ** 2 + aV[1] * xM[t - 2, 0]
        for im in range(1, k):
            # xM[t, im] = aV[0] - (
            #         0.5 * cV[im] * (xM[t - 1, im - 1] + xM[t - 1, im]) - (1 - cV[im]) * xM[t - 1, im]) ** 2 + aV[1] * \
            #             xM[t - 2, im]
            xM[t, im] = aV[0] - cV[im] * xM[t - 1, im - 1] * xM[t - 1, im] - (1 - cV[im]) * xM[t - 1, im] ** 2 + aV[1] * xM[t - 2, im]
    xM = xM[ntrans:n + ntrans, :]
    if check_if_unstable(xM):
        return None, None
    adjmatrix = np.zeros(shape=(k, k))
    for col in np.arange(k - 1):
        adjmatrix[col, col + 1] = 1
    return xM, adjmatrix


def coupledhenonmaps(k, c, n, aV=np.array([1.4, 0.3]), x0M=np.nan):
    '''
    xM = causalhenonmaps2(c,n,aV,x0V)
    COUPLEDHENONMAPS2 generates m time series from identical coupled Henon
    maps with coupling strength c. Each map is coupled to the next apart from
    the first and last map, the first only drives the second and the last
    only drives the second last (first and last are not coupled).
    INPUTS
    - K   : the number of coupled maps (time series)
    - c   : the coupling strength
    - n   : the time series length
    - aV  : the two parameters in a vector (if omitted the standard
            parameters for chaos are chosen)
    - x0M : initial conditions, a matrix of size 2 x m
    OUTPUTS
    - xM  : the n x m matrix of the generated time series
    '''

    aV = aV.reshape(-1, 1)
    if len(c) == 1:
        cV = c * np.ones((k, 1))
    else:
        cV = c
    if np.isnan(x0M):
        ntrans = 1000
        x0limV = np.array([-1, 1, - 0.3, 0.3]).reshape(-1, 1)
        xM = np.full((n + ntrans, k), np.nan)
        tmpM = np.random.uniform(0, 1, (2, k))
        xM[0, :] = (x0limV[3] - tmpM[0, :]) / (x0limV[3] - x0limV[2])
        xM[1, :] = (x0limV[1] - tmpM[1, :]) / (x0limV[1] - x0limV[0])
    else:
        ntrans = 2
        xM = np.full((n + ntrans, k), np.nan)
        xM[:2, :] = x0M
    for t in np.arange(2, n + ntrans):
        xM[t, 0] = aV[0] - xM[t - 1, 0] ** 2 + aV[1] * xM[t - 2, 0]
        for im in np.arange(1, k - 1):
            # xM[t, im] = aV[0] - (
            #         0.5 * cV[im] * (xM[t - 1, im - 1] + xM[t - 1, im + 1]) + (1 - cV[im]) * xM[t - 1, im]) ** 2 + \
            #             aV[1] * xM[t - 2, im]
            xM[t, im] = aV[0] - (
                    0.5 * cV[im] * (xM[t - 1, im - 1] + xM[t - 1, im + 1]) + (1 - cV[im]) * xM[t - 1, im]) ** 2 + \
                        aV[1] * xM[t - 2, im]

        xM[t, k - 1] = aV[0] - xM[t - 1, k - 1] ** 2 + aV[1] * xM[t - 2, k - 1]
    xM = xM[ntrans + 1:n + ntrans, :]
    if check_if_unstable(xM):
        return None, None
    adjmatrix = np.zeros(shape=(k, k))
    for col in np.arange(1, k - 1):
        adjmatrix[col - 1, col] = 1
        adjmatrix[col + 1, col] = 1
    return xM, adjmatrix


def coupledhenonmapssynergy(k, c, n, aV=np.array([1.4, 0.3]), x0M=np.nan):
    '''
    synergy on coupled henon maps, product instead of sum
    '''

    aV = aV.reshape(-1, 1)
    if len(c) == 1:
        cV = c * np.ones((k, 1))
    else:
        cV = c
    if np.isnan(x0M):
        ntrans = 1000
        x0limV = np.array([-1, 1, - 0.3, 0.3]).reshape(-1, 1)
        xM = np.full((n + ntrans, k), np.nan)
        tmpM = np.random.uniform(0, 1, (2, k))
        xM[0, :] = (x0limV[3] - tmpM[0, :]) / (x0limV[3] - x0limV[2])
        xM[1, :] = (x0limV[1] - tmpM[1, :]) / (x0limV[1] - x0limV[0])
    else:
        ntrans = 2
        xM = np.full((n + ntrans, k), np.nan)
        xM[:2, :] = x0M
    for t in np.arange(2, n + ntrans):
        xM[t, 0] = aV[0] - xM[t - 1, 0] ** 2 + aV[1] * xM[t - 2, 0]
        for im in np.arange(1, k - 1):
            xM[t, im] = aV[0] - (
                    0.5 * cV[im] * (xM[t - 1, im - 1] * xM[t - 1, im + 1]) + (1 - cV[im]) * xM[t - 1, im]) ** 2 + \
                        aV[1] * xM[t - 2, im]
        xM[t, k - 1] = aV[0] - xM[t - 1, k - 1] ** 2 + aV[1] * xM[t - 2, k - 1]
    xM = xM[ntrans + 1:n + ntrans, :]
    if check_if_unstable(xM):
        return None, None
    k = xM.shape[1]
    adjmatrix = np.zeros(shape=(k, k))
    for col in np.arange(1, k - 1):
        adjmatrix[col - 1, col] = 1
        adjmatrix[col + 1, col] = 1
    return xM, adjmatrix


def coupledmackeyglassflow(cM, tau=100, ts=4, data_size=2048):
    from jitcdde import jitcdde, y, t
    '''
    x_j'(t) = -gamma * x_j(t) + b * S (c_ij * x_i(t - tau)) / (1 + x_i(t - tau) ** n)
    i = 1,...,K
    j = 1,...,K
    n = 10
    gamma = 0.1
    tau: delta (17, 30, 100)
    ts: sampling time (1, 2, 4)
    '''
    # MG parameters
    n = 10
    b = 1.
    gamma = 0.1

    # #get K from coupling matrix
    d = cM.shape[0]
    consts = []
    # put in a array the constant terms of MG
    for i in np.arange(d):
        temp = -gamma * y(i, t)
        consts.append(temp)
    # put in array drivers terms
    drivers = []
    for i in np.arange(d):
        temp = y(i, t - tau) / (1 + y(i, t - tau) ** n)
        drivers.append(temp)

    # get the eqns
    def mg():
        for i in np.arange(d):
            temp = np.dot(drivers, cM[:, i]) + consts[i]
            yield temp

    print('setting system...')
    DDE = jitcdde(mg, n=d)
    # set random initial conditions
    DDE.constant_past(np.random.random(size=d))
    print('step on...')
    DDE.step_on_discontinuities()

    # set data array to store results
    data = []
    # run time to reach steady state
    t_trans = ts * 500
    # timesteps to sample time series
    t_sample = ts * data_size
    times = np.arange(t_trans + ts, t_trans + t_sample, ts)
    # for time in np.arange(DDE.t, DDE.t + data_size, ts):
    print('integrating...')
    for time in times:
        temp = DDE.integrate(time)
        data.append(temp)
        # print(DDE.integrate(time)[0])
    mg_coupled_system = np.zeros(shape=(len(data), d))
    # get data into results
    for each in np.arange(len(data)):
        for value in np.arange(len(data[each])):
            mg_coupled_system[each, value] = data[each][value]
    if check_if_unstable(mg_coupled_system):
        return None, None
    return mg_coupled_system, cM
