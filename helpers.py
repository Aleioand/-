import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import networkx as nx
from scipy.optimize import linprog
from scipy import stats
from math import factorial
from sklearn.neighbors import KDTree
from scipy.special import psi

tex_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 14,
    "font.size": 16,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
}

plt.rcParams.update(tex_fonts)


def mvr_exponential(rho=0., size=1000):
    mvr_norm = stats.multivariate_normal([0, 0], [[1., rho], [rho, 1.]])
    mvr_norm_data = mvr_norm.rvs(size=size)
    # sns.jointplot(mvr_norm_data[:, 0], mvr_norm_data[:, 1], kind='kde')
    x = stats.expon()
    y = stats.expon()
    norm = stats.norm()
    x_unif = norm.cdf(mvr_norm_data)
    x_trans = x.ppf(x_unif[:, 0])
    y_trans = y.ppf(x_unif[:, 1])
    # sns.jointplot(x_trans, y_trans)
    data = np.concatenate([x_trans.reshape(-1, 1), y_trans.reshape(-1, 1)], axis=1)
    return data

def truncated_normal(a=-2, b=2, loc=0, scale=1, size=1000):
    trunc = stats.truncnorm(a=a, b=b, loc=loc, scale=scale)
    data = trunc.rvs(size)
    # sns.distplot(data)
    return data

def in_hull(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    try:
        A = np.r_[points.T, np.ones((1, n_points))]
        b = np.r_[x, np.ones(1)]
    except:
        return False
    try:
        lp = linprog(c, A_eq=A, b_eq=b)
    except:
        return False
    return lp.success

def nrmse(correct_values, predicted):
    # mean_corr = np.mean(correct_values)
    # var_corr = np.sum((correct_values - mean_corr) ** 2)
    # error = correct_values - predicted
    # error.clip(lower=error.quantile(q=0.01),upper=error.quantile(q=0.99),inplace=True)
    # var_pred = np.sum(error ** 2)
    # value = np.sqrt(var_pred / var_corr)
    error = predicted - correct_values
    value = np.sqrt(np.sum(error ** 2) / len(error)) / np.std(correct_values)
    return value

def calculate_measures(real, estimated):
    c_m = real.copy()
    r_m = estimated.copy()
    np.fill_diagonal(c_m, 0.0)
    np.fill_diagonal(r_m, 0.0)
    # c_m = c_m.flatten()
    # r_m = r_m.flatten()
    c_m[c_m > 0] = 1
    r_m[r_m > 0] = 1
    ad_m = c_m + r_m
    sub_m = c_m - r_m
    tp = len(ad_m[ad_m == 2])
    tn = len(ad_m[ad_m == 0])
    fp = len(sub_m[sub_m == -1])
    fn = len(sub_m[sub_m == 1])

    # hd = how many pairs are misclassified
    hamming = np.count_nonzero(np.subtract(c_m, r_m))
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)  # true positive rate
    recall = tp / (tp + fn)  # sensitivity (how many connections finds)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)  # how many non-connections finds
    if precision == 0 and recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)
    # mcc = 1 (perfect), mcc = -1 (total disagreement), mcc = 0 (random assignments)
    if precision == 0.:
        mcc = 0
    else:
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(precision=precision, recall=recall, accuracy=accuracy, f_score=f_score, mcc=mcc, hamming=hamming,
                specificity=specificity)

def gaussianisation(data):
    sort_ind = np.argsort(data)
    gaussian_data = np.random.normal(0, 1, size=data.shape[0])
    gaussian_data_ind = np.argsort(gaussian_data)
    g_d_sorted = gaussian_data[gaussian_data_ind]
    y = np.zeros(shape=data.shape[0])
    for i in np.arange(data.shape[0]):
        y[i] = g_d_sorted[sort_ind[i]]
    return y

def create_causality_network(connectivity_matrix, original_causal_direction=True):
    # from the connectivity matrix, create the directional graph
    g = nx.DiGraph()
    np.fill_diagonal(connectivity_matrix, 0.0)
    k = connectivity_matrix.shape[0]
    nodes = np.arange(k)
    g.add_nodes_from(nodes)
    # df = pd.DataFrame(connectivity_matrix,index=[str(i) for i in np.arange(1,k+1)],columns=[str(i) for i in np.arange(1,k+1)])
    # edges = [(start,end) for start in df.index for end in df.columns if np.logical_and(df.loc[start,end] != 0, start != end) ]
    # edges = [(start + 1, end + 1) for start in np.arange(k) for end in np.arange(k) if connectivity_matrix[start, end] > 0]
    edges = []
    if original_causal_direction:
        change_direction_threshold = - 1
    else:
        change_direction_threshold = 0.5
    for start in np.arange(k):
        for end in np.arange(k):
            if connectivity_matrix[start, end] > 0:
                if np.random.random() > change_direction_threshold:
                    edges.append((start, end))
                else:
                    edges.append((end, start))

    # edges = [(start, end) for start in np.arange(k) for end in np.arange(k) if connectivity_matrix[start, end] > 0]
    g.add_edges_from(edges)
    nodes_relabelling = {i:i+1 for i in np.arange(k)}
    g = nx.relabel_nodes(g, nodes_relabelling)
    return g

def plot_causality_network(graphs, titles=[], savepath='', nodelist=[], suptitle=None):
    '''
    plot network given graph
    '''
    assert type(graphs) == list, 'Graphs should be a list'
    nr_plot_y = len(graphs)
    # number_plot = np.int(np.ceil(np.sqrt(len(graphs))))
    plt.close('all')
    if len(titles) < 2:
        fig = plt.figure(figsize=(4, 4))
    else:
        fig = plt.figure(figsize=(8, 4))
    # fig, axes = plt.subplots(number_plot, number_plot, num=1)
    for i, graph in enumerate(graphs):
        ax = fig.add_subplot(1, nr_plot_y, i + 1)
        # ix = np.unravel_index(i, axes.shape)
        # get no of nodes
        k = graph.number_of_nodes()
        # #choose layout
        # pos = nx.spring_layout(graph)
        # pos = nx.random_layout(graph)
        pos = nx.shell_layout(graph)
        # pos = nx.circular_layout(graph, scale=1.1)
        # plt.sca(axes[ix])
        node_colors = ['y']*k
        for each_comm in nodelist:
            if len(nodelist) == 0:
                random_color = 'y'
            else:
                random_color = np.random.choice(a=['blue', 'red', 'green', 'cyan', 'magenta'])
            node_colors.extend([random_color] * len(each_comm))


        # # # ###move nodes broader
        # # Draw the nodes and edges
        # nx.draw_networkx_nodes(graph, pos, node_size=100, node_color=node_colors )
        # nx.draw_networkx_edges(graph, pos, arrowsize=12, connectionstyle='arc3,rad=0.15')
        # # Create a dictionary for the modified positions
        # new_pos = {}
        # # Get the maximum x and y values
        # max_x = max(x for x, y in pos.values())
        # max_y = max(y for x, y in pos.values())
        #
        # # Iterate over all the positions
        # for node, (x, y) in pos.items():
        #     # Multiply the x and y values by 1.5 to move the labels further out
        #     new_pos[node] = (1.1 * x, 1.1 * y)
        # # Draw the labels using the modified positions
        # nx.draw_networkx_labels(graph, new_pos, font_size=12, )


        nodes = nx.draw_networkx_nodes(graph, pos, node_size=300, node_color=node_colors,  )
        nodes.set_edgecolor('b')
        nx.draw_networkx_labels(graph, pos, font_size=12, )
        nx.draw_networkx_edges(graph, pos, arrowsize=12, connectionstyle='arc3,rad=0.15')


        # if k > 99:
        #     nx.draw(graph, pos=pos, ax=ax, with_labels=True, node_size=300, font_size=6, node_color=node_colors, arrowsize=8, )
        # else:
        #     nx.draw(graph, pos=pos, ax=ax, with_labels=True, node_size=300, font_size=12, node_color=node_colors, arrowsize=12, connectionstyle='arc3,rad=0.15')
        if len(titles) > 0:
            ax.set_title(titles[i])
        else:
            ax.set_title('')
        ax.set_axis_off()
    if suptitle is not None:
        plt.suptitle(suptitle)
    # save the drawing
    if savepath != '':
        # Using seaborn's style
        plt.style.use('seaborn')
        # With LaTex fonts
        # use LaTeX fonts in the plot
        # plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        if 'png' in savepath:
            plt.savefig(f'{savepath}', bbox_inches='tight')
        else:
            plt.savefig(f'{savepath}', bbox_inches='tight', format='pdf')
    return

def plot_mv_timeseries(xM, figsize=(8, 6), save_path=None, show=False, color='k'):
    fig, axes = plt.subplots(nrows=xM.shape[1], ncols=1, figsize=figsize)
    fig.text(0.5, 0.08, 't', va='center',)
    fig.text(0.08, 0.5, 'x(t)', va='center', rotation='vertical')
    for i in np.arange(axes.shape[0]):
        axes[i].plot(xM[:, i], color=color)
        axes[i].set_yticklabels([])
        if i < xM.shape[0] - 1:
            axes[i].set_xticklabels([])
    plt.subplots_adjust(hspace=0)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig

def plot_mv_timeseries_25(xM, savepath=None, show=False):
    '''
    plot timeseries in chunks of 25
    '''
    n, d = xM.shape
    ts_chunks = 25
    if d > 50:
        ts_chunks = 50
    n_graphs = (d - 1) // ts_chunks + 1
    n_cols = (n_graphs + 1) // 2
    fig = plt.figure(figsize=(8, 10))
    xticklabels = (np.arange(0, n, step=np.int(0.2*n))//100)*100
    fig.text(0.1, 0.5, 'x(t)', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.04, 't', fontsize=16)
    n_inner_grid = 1
    if n_graphs > 1:
        n_inner_grid = 2
        if n_cols > 1:
            fig.text(0.5, 0.5, 'x(t)', rotation='vertical', fontsize=14)
            xticklabels = (np.arange(0, n, step=np.int(0.4 * n)) // 100) * 100
    outer = gridspec.GridSpec(1, n_cols, wspace=0.2, hspace=0.2)
    for i in np.arange(n_cols):
        inner = gridspec.GridSpecFromSubplotSpec(n_inner_grid, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.01)
        for its in np.arange(n_inner_grid):
            ts = gridspec.GridSpecFromSubplotSpec(ts_chunks, 1, subplot_spec=inner[its], wspace=0.1, hspace=0.01)
            for j in np.arange(ts_chunks):
                ax = fig.add_subplot(ts[j])
                ax.plot(xM[:, j*i+j])
                ax.tick_params(axis='y', length=0)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                if its == n_inner_grid - 1 and j == ts_chunks - 1:
                    ax.set_xticklabels(xticklabels, fontsize=14)
                    ax.set_xticks(xticklabels)
                fig.add_subplot(ax)
    if savepath is not None:
        # Using seaborn's style
        plt.style.use('seaborn')
        # With LaTex fonts
        # use LaTeX fonts in the plot
        # plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.savefig(f'{savepath}', bbox_inches='tight', format='pdf')
    if show:
        plt.show()
    return fig


def plot_cmap(rm, save_path=None):
    d = rm.shape[0]
    fig, ax = plt.subplots(1, 1)
    ax.imshow(rm, origin='lower', cmap='gray')
    ax.set_xticks(np.arange(0, d, 2))
    ax.set_yticks(np.arange(0, d, 2))
    for i in range(d):
        for j in range(d):
            if i == j:
                ax.text(i, j, 'x', ha='center', va='center', color='white')
    if save_path is not None:
        plt.savefig(save_path)
    return fig

def example_plots():
    from sklearn.linear_model import LinearRegression, Ridge
    data = np.array([[1, 1], [2, 2]])
    X = data[:, 0]
    y = data[:, 1]
    linear_model = LinearRegression().fit(X=X.reshape(-1, 1), y=y)
    rr_model_01 = Ridge(alpha=0.1).fit(X=X.reshape(-1, 1), y=y)
    rr_model_05 = Ridge(alpha=0.5).fit(X=X.reshape(-1, 1), y=y)
    rr_model_1 = Ridge(alpha=1.0).fit(X=X.reshape(-1, 1), y=y)

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(X, y, marker='o', color='black', )
    plt.plot(np.arange(0.5, 3.5), linear_model.predict(np.arange(0.5, 3.5).reshape(-1, 1)), label='OLS')
    plt.plot(np.arange(0.5, 3.5), rr_model_01.predict(np.arange(0.5, 3.5).reshape(-1, 1)), label='RR(alpha = 0.1)')
    plt.plot(np.arange(0.5, 3.5), rr_model_05.predict(np.arange(0.5, 3.5).reshape(-1, 1)), label='RR(alpha = 0.5)')
    plt.plot(np.arange(0.5, 3.5), rr_model_1.predict(np.arange(0.5, 3.5).reshape(-1, 1)), label='RR(alpha = 1.0)')
    plt.legend()

def get_adj_matrix_from_graph(G):
    adjacency_matrix = np.zeros(shape=(len(G.nodes()), len(G.nodes())))
    for i in G.edges():
        adjacency_matrix[i[0], i[1]] = 1
    return adjacency_matrix

def k_distrib(graph, scale='lin', colour='#40a6d1', alpha=.8, expct_lo=1, expct_hi=10, expct_const=1):
    plt.close()
    num_nodes = graph.number_of_nodes()
    max_degree = 0
    # Calculate the maximum degree to know the range of x-axis
    for n in graph.nodes():
        if graph.degree(n) > max_degree:
            max_degree = graph.degree(n)
    # X-axis and y-axis values
    x = []
    y_tmp = []
    # loop for all degrees until the maximum to compute the portion of nodes for that degree
    for i in range(max_degree + 1):
        x.append(i)
        y_tmp.append(0)
        for n in graph.nodes():
            if graph.degree(n) == i:
                y_tmp[i] += 1
        y = [i / num_nodes for i in y_tmp]
    # Plot the graph
    deg, = plt.plot(x, y, label='Degree distribution', linewidth=0, marker='o', markersize=8, color=colour, alpha=alpha)
    # Check for the lin / log parameter and set axes scale
    if scale == 'log':
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Degree distribution (log-log scale)')
        # add theoretical distribution line k^-3
        w = [a for a in range(expct_lo, expct_hi)]
        z = []
        for i in w:
            x = (i ** -3) * expct_const  # set line's length and fit intercept
            z.append(x)

        plt.plot(w, z, 'k-', color='#7f7f7f')
    else:
        plt.title('Degree distribution (linear scale)')

    plt.ylabel('P(k)')
    plt.xlabel('k')
    plt.show()

def theoretical_entropy_normal(rho):
    return 0.5 * np.log(np.power(2 * np.pi * np.e, rho.shape[0]) * np.linalg.det(rho))

def theoretical_MI(rho_joint, rho_x, rho_y):
    h_x = theoretical_entropy_normal(rho=rho_x)
    h_y = theoretical_entropy_normal(rho=rho_y)
    h_xy = theoretical_entropy_normal(rho=rho_joint)
    return h_x + h_y - h_xy

def cube_transformation(data):
    cuber = lambda x: x ** 3
    result = np.array([cuber(i) for i in data])
    return result

def exp_transformation(data):
    to_exp = lambda x: np.exp(x)
    result = np.array([to_exp(i) for i in data])
    return result

def theoretical_entropy_data(data):
    from math import pi, e
    if data.shape[1] > 1:
        return 0.5 * np.log(np.linalg.det(2 * pi * e * ((np.dot(data.T, data)) / (data.shape[0] - 1))))
        # return 0.5 * np.log(np.linalg.det(2 * pi * e * np.cov(data.T)))
    else:
        return 0.5 * np.log(2 * pi * e * np.var(data.T))

def theoretical_mutual_info_normal_data(x, y):
    h_x = theoretical_entropy_data(x)
    h_y = theoretical_entropy_data(y)
    try:
        dim_x = x.shape[1]
    except:
        dim_x = 1
    try:
        dim_y = y.shape[1]
    except:
        dim_y = 1
    h_xy = theoretical_entropy_data(np.concatenate([x.reshape(-1, dim_x), y.reshape(-1, dim_y)], axis=1))
    print(f'theoretical entropy of variable x = {h_x}')
    print(f'theoretical entropy of variable y = {h_y}')
    print(f'theoretical entropy of joint variable x-y = {h_xy}')
    return h_x + h_y - h_xy

def mutual_info_estimator_kraskov_rect(x, y, metric='chebyshev', k=5):
    from scipy.special import psi
    if x.shape[0] != y.shape[0]:
        print('Variables have unequal size!')
    # find variables dimension
    try:
        dim_x = x.shape[1]
    except:
        dim_x = 1
    try:
        dim_y = y.shape[1]
    except:
        dim_y = 1

    # normalize data into [0,1]
    # x_scaled = (x - np.min(x, axis=0)) / (np.ptp(x, axis=0))
    # y_scaled = (y - np.min(y, axis=0)) / (np.ptp(y, axis=0))
    x_scaled = x.copy()
    y_scaled = y.copy()
    # make joint dataframe with marginal variables
    df = np.concatenate((x_scaled.reshape(-1, dim_x), y_scaled.reshape(-1, dim_y)), axis=1)  # dataset
    N = df.shape[0]
    psiK = psi(k)
    psiN = psi(N)

    # Kraskov MI estimation
    neighbors_joint, distsM = ANN(df, k + 1)
    max_dist_marginal = np.zeros(shape=(N, dim_x + dim_y))
    for point in neighbors_joint:
        # get local points
        local_points = df[point]
        # set ref point into origin
        local_points_centered = local_points - local_points[0]
        # get max dist in each marginal
        max_dists = np.max(np.abs(local_points_centered), axis=0)
        max_dist_marginal[point[0], :] = max_dists
    nX = nneighforgivenr(X=x_scaled.reshape(-1, dim_x), rV=max_dist_marginal[:, 0] - np.ones(N) * 10 ** (-10))
    nY = nneighforgivenr(X=y_scaled.reshape(-1, dim_y), rV=max_dist_marginal[:, 1] - np.ones(N) * 10 ** (-10))
    psibothM = psi(np.concatenate((nX.reshape(N, 1), nY.reshape(N, 1)), axis=1))
    mi_kraskov_rect = psiK + (dim_x + dim_y - 1) * psiN - (dim_x + dim_y - 1) / k - np.mean(np.sum(psibothM,
                                                                                                   axis=1))  # I(X;Y) = ψ(k) + (d - 1) * ψ(Ν) - (d - 1)/k - <ψ(Nx + 1) + ψ(Ny + 1)>, original estimator
    return mi_kraskov_rect, neighbors_joint, max_dist_marginal

def entropy_estimator(x, k=5, base=np.exp(1)):
    '''
    K-L k-nearest neighbor continuous entropy estimator
        x: list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]] if x is a one-dimensional scalar and we have four samples
    '''

    from scipy.spatial import cKDTree
    from scipy.special import digamma
    # assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    # get X dimension
    d = len(x[0])
    # get X size
    N = len(x)
    # insert small noise into the data
    x = [list(p + 10e-10 * np.random.rand(d)) for p in x]

    # build tree
    tree = cKDTree(x)
    # find knn neighbors
    nn = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in x]

    # calculate entropy with KL estimator
    # H(x) = psi(N) - psi(k) + log(Cd) + d * <log(e)>, e: twice the distance from Xi to its k-th neighbor, cd: max-norm = 1,
    const = digamma(N) - digamma(k) + d * np.log(2)

    return (const + d * np.mean(list(map(np.log, nn)))) / np.log(base)

def create_coupling_matrix(a_m=None, autostrength=0.5):
    '''
    create coupling matrix given an adjacency matrix am
    autostrength: coupling strength of time series itself
    '''
    if a_m is None:
        a_m = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 1, 1, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 1, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0, 1]
                        ]).T
    k = a_m.shape[0]
    c_m = np.random.uniform(0, 1, size=(k, k))
    for k_ in np.arange(k):
        a_v = np.multiply(c_m[:, k_], a_m[:, k_])
        idx_v = np.setdiff1d(np.arange(k), k_)
        saa = np.sum(a_v[idx_v])
        if saa > 0:  # normalize in order sum(c_m[:,k_])=1
            c_m[idx_v, k_] = (1 - autostrength) * a_v[idx_v] / saa
            c_m[k_, k_] = autostrength
        else:
            c_m[idx_v, k_] = np.zeros(shape=len(idx_v))
            c_m[k_, k_] = 1.
    return np.round(c_m, 3)

def permutation_entropy(time_series, order=3, delay=1, normalize=False):
    """Permutation Entropy.
    Parameters
    ----------
    time_series : list or np.array
        Time series
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1. Otherwise, return the permutation entropy in bit.
    Returns
    -------
    pe : float
        Permutation Entropy
    References
    ----------
    .. [1] Massimiliano Zanin et al. Permutation Entropy and Its Main
        Biomedical and Econophysics Applications: A Review.
        http://www.mdpi.com/1099-4300/14/8/1553/pdf
    .. [2] Christoph Bandt and Bernd Pompe. Permutation entropy — a natural
        complexity measure for time series.
        http://stubber.math-inf.uni-greifswald.de/pub/full/prep/2001/11.pdf
    Notes
    -----
    Last updated (Oct 2018) by Raphael Vallat (raphaelvallat9@gmail.com):
    - Major speed improvements
    - Use of base 2 instead of base e
    - Added normalization
    Examples
    --------
    1. Permutation entropy with order 2
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value between 0 and log2(factorial(order))
        >>> print(permutation_entropy(x, order=2))
            0.918
    2. Normalized permutation entropy with order 3
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(permutation_entropy(x, order=3, normalize=True))
            0.589
    """
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe

def _embed(x, order=3, delay=1):
    """Time-delay embedding.
    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.
    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

def weighted_pe(xM, embdim, embdelay, normalize=False):
    import itertools
    from math import factorial
    temp_list = list()
    wop = list()
    for i in range(len(xM) - embdelay * (embdim - 1)):
        Xi = xM[i:(embdim + i)]
        Xn = xM[(i + embdim - 1): (i + embdim + embdim - 1)]
        Xi_mean = np.mean(Xi)
        Xi_var = (Xi - Xi_mean) ** 2
        weight = np.mean(Xi_var)
        sorted_index_array = list(np.argsort(Xi))
        temp_list.append([''.join(map(str, sorted_index_array)), weight])
    result = pd.DataFrame(temp_list, columns=['pattern', 'weights'])
    for pat in (result['pattern'].unique()):
        wop.append(np.sum(result.loc[result['pattern'] == pat, 'weights'].values))
    temp = wop / sum(wop)
    wpe = -np.multiply(temp, np.log2(temp)).sum()
    if normalize:
        wpe /= np.log2(factorial(embdim))
    return (wpe)

def mi_estimator_from_projection(xV, yV, zM, nnei=5):
    '''
    calculates I(X;Y) after projection space (X,Y,Z)
    # Ip(X;Y) = ψ(Ν) - <ψ(Nx + 1) + ψ(Ny + 1) - ψ(Nxy + 1) >
    '''
    from scipy.special import psi

    n = xV.shape[0]
    psi_n = psi(n)

    xall_M = np.concatenate((xV, yV, zM), axis=1)
    _, distsM = ANN(xall_M, nnei + 1)
    maxdistV = distsM[:, -1]
    n_x = nneighforgivenr(X=xV, rV=maxdistV - np.ones(n) * 10 ** (-10))
    n_y = nneighforgivenr(X=yV, rV=maxdistV - np.ones(n) * 10 ** (-10))
    n_xy = nneighforgivenr(X=np.concatenate((xV, yV), axis=1), rV=maxdistV - np.ones(n) * 10 ** (-10))

    psi_mar = np.full((n, 3), np.nan)
    psi_mar[:, 0] = psi(n_x)
    psi_mar[:, 1] = psi(n_y)
    psi_mar[:, 2] = -psi(n_xy)

    mix_y = psi_n - np.mean(np.sum(np.concatenate(
        [psi(n_x).reshape(-1, 1),
         psi(n_y).reshape(-1, 1),
         -psi(n_xy).reshape(-1, 1)], axis=1
    ), axis=1))

    return mix_y

def cmi_estimator_1(xV, yV, zM, nnei=5, normalize=False):
  '''
  calculates I(X;Y | Z)
  # I(X;Y|Z) = ψ(k) - <ψ(Nxz + 1) + ψ(Nyz + 1) - ψ(Nz + 1)>,
  # I(X;Y|Z) = I(X;(Y,Z)) - I(X;Z)
  # I(X;(Y,Z)) = ψ(k) + ψ(Ν) - <ψ(Nx + 1) + ψ(Nyz + 1)>,
  # Ip(X;Z) = ψ(Ν) - <ψ(Nx + 1) + ψ(Nz + 1) - ψ(Nxz + 1) >

  '''
  from scipy.special import psi

  n = xV.shape[0]
  psi_n = psi(n)
  psi_nnei = psi(nnei)

  xall_M = np.concatenate((xV, yV, zM), axis=1)
  _, distsM = ANN(xall_M, nnei + 1)
  maxdistV = distsM[:, -1]
  n_z = nneighforgivenr(X=zM, rV=maxdistV - np.ones((n)) * 10 ** (-10))
  n_yz = nneighforgivenr(X=np.concatenate((yV, zM), axis=1), rV=maxdistV - np.ones((n)) * 10 ** (-10))
  n_xz = nneighforgivenr(X=np.concatenate((xV, zM), axis=1), rV=maxdistV - np.ones((n)) * 10 ** (-10))

  psi_mar = np.full((n, 3), np.nan)
  psi_mar[:, 0] = psi(n_xz)
  psi_mar[:, 1] = psi(n_yz)
  psi_mar[:, 2] = -psi(n_z)

  n_x = nneighforgivenr(X=xV, rV=maxdistV - np.ones((n)) * 10 ** (-10))

  psiff = psi(np.concatenate([n_x.reshape(-1, 1), n_yz.reshape(-1, 1)], axis=1))
  mix_yz = psi_nnei + psi_n - np.mean(np.sum(psiff, axis=1))
  mix_z = psi_n - np.mean(np.sum(np.concatenate(
      [-psi(n_xz).reshape(-1, 1),
       psi(n_x).reshape(-1, 1),
       psi(n_z).reshape(-1, 1)], axis=1
  ), axis=1))

  # cmi_xy_z = psi_nnei - np.mean(np.sum(psi_mar, axis=1))
  cmi_xy_z = mix_yz - mix_z
  return cmi_xy_z, nnei, np.mean(maxdistV)

def cmi_estimator_2(xV, yV, zM, nnei=5, normalize=False, add_noise=True):
    '''
    calculates I(X;Y | Z)
    # I(X;Y|Z) = ψ(k) - <ψ(Nxz + 1) + ψ(Nyz + 1) - ψ(Nz + 1)>,
    # I(X;Y|Z) = I(X;(Y,Z)) - I(X;Z)
    # I(X;(Y,Z)) = ψ(k) + ψ(Ν) - <ψ(Nx + 1) + ψ(Nyz + 1)>,
    # Ip(X;Z) = ψ(Ν) - <ψ(Nx + 1) + ψ(Nz + 1) - ψ(Nxz + 1) >
    # #rect CMI
    # # I(X;Y|Z) = ψ(k) - (d_x + d_y + d_z - 1)/k + <-ψ(Nxz + 1) - ψ(Nyz + 1) + ψ(Nz + 1) + (d_x + d_z - 1)/n_xz + (d_y + d_z - 1)/n_yz - + (d_z - 1)/n_z >,
    '''
    from scipy.special import psi
    from ANN import ANN, find_n_on_marginal

    n, dim_x = xV.shape
    n, dim_y = yV.shape
    n, dim_z = zM.shape

    # psi_n = psi(n)
    psi_nnei = psi(nnei)

    # #add small noise to data to avoid degeneracies (kraskov implementation details)
    xV = xV + 10e-10 * np.random.rand(1)[0]
    yV = yV + 10e-10 * np.random.rand(1)[0]
    zM = zM + 10e-10 * np.random.rand(1)[0]

    xall_M = np.concatenate((xV, yV, zM), axis=1)
    neighbors, distsM = ANN(xall_M, nnei + 1)
    maxdistM = np.full(shape=(n, dim_x + dim_y + dim_z), fill_value=-np.inf)

    for each_point in neighbors:
        current_point_idx = each_point[0]
        current_point = xall_M[current_point_idx]
        for i in range(nnei + 1):
            neighbor_idx = each_point[i]
            neighbor = xall_M[neighbor_idx]
            for j in range(dim_x + dim_y + dim_z):
                if maxdistM[current_point_idx][j] < abs(neighbor[j] - current_point[j]):
                    maxdistM[current_point_idx][j] = abs(neighbor[j] - current_point[j])

    # maxdistM = maxdistM - 1e-15
    maxdist_z = maxdistM[:, dim_x + dim_y: ]
    n_z = np.zeros(shape=n)

    for i in range(zM.shape[0]):
        temp_ = np.abs(zM[i] - zM) <= maxdist_z[i]
        counter = np.sum(np.sum(temp_, axis=1) == dim_z)
        n_z[i] = counter

    maxdist_yz = maxdistM[:, dim_x: ]
    yz = np.concatenate((yV, zM), axis=1)
    n_yz = np.zeros(shape=n)

    for i in range(yz.shape[0]):
        temp_ = np.abs(yz[i] - yz) <= maxdist_yz[i]
        counter = np.sum(np.sum(temp_, axis=1) == dim_y + dim_z)
        n_yz[i] = counter

    # maxdist_xz = np.delete(maxdistM, np.arange(dim_x, dim_x + dim_y), axis=1)
    maxdist_xz = np.concatenate((maxdistM[:, :dim_x], maxdistM[:, dim_x + dim_y:]), axis=1)
    xz = np.concatenate((xV, zM), axis=1)
    n_xz = np.zeros(shape=n)

    for i in range(n):
        temp_ = np.abs(xz[i] - xz) <= maxdist_xz[i]
        counter = np.sum(np.sum(temp_, axis=1) == dim_x + dim_z)
        n_xz[i] = counter

    n_xz = n_xz.reshape(n, -1) - 1
    n_yz = n_yz.reshape(n, -1) - 1
    n_z = n_z.reshape(n, -1) - 1
    psi_mar = np.full((n, 6), np.nan)
    psi_mar[:, [0]] = -psi(n_xz)
    psi_mar[:, [1]] = -psi(n_yz)
    psi_mar[:, [2]] = psi(n_z)
    psi_mar[:, [3]] = (dim_x + dim_z - 1) / n_xz
    psi_mar[:, [4]] = (dim_y + dim_z - 1) / n_yz
    psi_mar[:, [5]] = -(dim_z - 1) / n_z
    # maxdist_x = maxdistM[:, :dim_x]
    # n_x = np.zeros(shape=zM.shape[0])
    # for i in range(xV.shape[0]):
    #     temp_ = np.abs(xV[i] - xV) < maxdist_x[i]
    #     counter = np.sum(np.sum(temp_, axis=1) == dim_x)
    #     n_x[i] = counter
    # psiff = psi(np.concatenate([n_x.reshape(-1, 1), n_yz.reshape(-1, 1)], axis=1))
    # mix_yz = psi_nnei + psi_n - np.mean(np.sum(psiff, axis=1))
    # mix_z = psi_n - np.mean(np.sum(np.concatenate(
    #     [-psi(n_xz).reshape(-1, 1),
    #      psi(n_x).reshape(-1, 1),
    #      psi(n_z).reshape(-1, 1)], axis=1
    # ), axis=1))
    # cmi_xy_z = mix_yz - mix_z
    # #https://beteje.github.io/assets/pdf/2017_PRE.pdf
    cmi_xy_z = psi_nnei - (dim_x + dim_y + dim_z - 1) / nnei + np.mean(np.sum(psi_mar, axis=1))
    return cmi_xy_z

def mi_estimator_ksg1(xV, yV, nnei=5, normalize=False):
  '''
  calculates I(X;Y) using KSG algorithm1 (with max-norms squares)
  '''
  from scipy.special import psi

  n = xV.shape[0]
  psi_nnei = psi(nnei)
  psi_n = psi(n)

  if normalize:
      xV = (xV - np.min(xV)) / np.ptp(xV)
      yV = (yV - np.min(yV)) / np.ptp(yV)

  xembM = np.concatenate((xV, yV), axis=1)
  _, distsM = ANN(xembM, nnei + 1)
  maxdistV = distsM[:, -1]
  n_x = nneighforgivenr(X=xV, rV=maxdistV - np.ones(n) * 10 ** (-10))
  n_y = nneighforgivenr(X=yV, rV=maxdistV - np.ones(n) * 10 ** (-10))
  psibothM = psi(np.concatenate((n_x.reshape(-1, 1), n_y.reshape(-1, 1)), axis=1))
  #     # I(X;Y) = ψ(k) + ψ(Ν) - <ψ(Nx + 1) + ψ(Ny + 1)>
  mi = psi_nnei + psi_n - np.mean(np.sum(psibothM, axis=1))
  return mi, nnei, np.mean(maxdistV)

def mi_estimator_ksg2(xV, yV, nnei=5, normalize=False):
    '''
    calculates I(X;Y) using KSG algorithm1 rectangular approach)
    '''
    from scipy.special import psi
    from ANN import ANN, find_n_on_marginal

    n, dim_x = xV.shape
    n, dim_y = yV.shape

    psi_nnei = psi(nnei)
    psi_n = psi(n)
    if normalize:
        xV = (xV - np.min(xV)) / np.ptp(xV)
        yV = (yV - np.min(yV)) / np.ptp(yV)

    # #Kraskov et. al. in the "Implementation details" section describe adding low intensity
    # noise to break degeneracies in calculating nearest neighbors
    xV = xV + 10e-10 * np.random.rand(1)[0]
    yV = yV + 10e-10 * np.random.rand(1)[0]

    xembM = np.concatenate((xV, yV), axis=1)
    neighbors, distsM = ANN(xembM, nnei + 1)
    maxdistM = np.full(shape=(xembM.shape[0], xembM.shape[1]), fill_value=-np.inf)

    for each_point in neighbors:
        current_point_idx = each_point[0]
        current_point = xembM[current_point_idx]
        for i in range(nnei + 1):
            neighbor_idx = each_point[i]
            neighbor = xembM[neighbor_idx]
            for j in range(dim_x + dim_y):
                if maxdistM[current_point_idx][j] < abs(neighbor[j] - current_point[j]):
                    maxdistM[current_point_idx][j] = abs(neighbor[j] - current_point[j])

    # #plots
    # import matplotlib.patches as patches
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # # ax.scatter(xembM[:, 0], xembM[:, 1])
    # local_neigh = np.array(each_point)
    # ax.scatter(xembM[local_neigh][:, 0], xembM[local_neigh][:, 1])
    # ax.scatter(xembM[local_neigh[0]][0], xembM[local_neigh[0]][1])
    # x_sq_bottom_left = xembM[local_neigh[0]][0] - distsM[current_point_idx, -1]
    # y_sq_bottom_left = xembM[local_neigh[0]][1] - distsM[current_point_idx, -1]
    # area_sq = patches.Rectangle((x_sq_bottom_left, y_sq_bottom_left),
    #                          width=2 * distsM[current_point_idx, -1], height=2 * distsM[current_point_idx, -1], linewidth=1, edgecolor='r',
    #                          facecolor='none')
    # ax.add_patch(area_sq)
    # area_rect = patches.Rectangle(((current_point[0] - maxdistM[current_point_idx, 0]), (current_point[1] - maxdistM[current_point_idx, 1])),
    #                          width=2 * maxdistM[current_point_idx, 0], height=2 * maxdistM[current_point_idx, 1], linewidth=1, edgecolor='g',
    #                          facecolor='none')
    # ax.add_patch(area_rect)
    # plt.ylim(ymax=current_point[1] + distsM[current_point_idx, -1] + 0.05, ymin=current_point[1] - distsM[current_point_idx, -1]  - 0.05 )
    # plt.xlim(xmax=current_point[0] + distsM[current_point_idx, -1] + 0.05, xmin=current_point[0] - distsM[current_point_idx, -1] - 0.05 )
    # plt.tight_layout()
    # plt.show()

    maxdistM = maxdistM - 1e-15
    n_x = find_n_on_marginal(marginal=xV, maxdist=maxdistM[:, :dim_x], add_noise=False, square=False)
    n_y = find_n_on_marginal(marginal=yV, maxdist=maxdistM[:, dim_x:], add_noise=False, square=False)
    n_x = n_x.reshape(n, -1)
    n_y = n_y.reshape(n, -1)
    # psibothM = psi(np.concatenate((n_x.reshape(-1, 1), n_y.reshape(-1, 1)), axis=1))
    # mi = psi_nnei + psi_n - np.mean(np.sum(psibothM, axis=1)) - 1 / nnei
    #     # I(X;Y) = ψ(k) + ψ(Ν) - <ψ(Nx) + ψ(Ny)> - 1/k
    #     # I(X;Y) = ψ(k) + ψ(Ν) - <ψ(Nx) + ψ(Ny) - ((d_x -1)/n_x ) - ((d_y -1)/n_y) > - ((d_x + d_y - 1)/k )
    psi_all = np.full(shape=(n, 4), fill_value=np.nan)
    psi_all[:, [0]] = psi(n_x)
    psi_all[:, [1]] = psi(n_y)
    psi_all[:, [2]] = -(dim_x - 1)/n_x
    psi_all[:, [3]] = -(dim_y - 1)/n_y
    mi = psi_nnei + psi_n - np.mean(np.sum(psi_all, axis=1)) - (dim_x + dim_y - 1) / nnei
    return mi

def ANN(X, k):
    tree = KDTree(X, leaf_size=1, metric='chebyshev')
    dists, nnidx = tree.query(X, k=k)
    del tree
    return nnidx, dists

def ANNR(X, rV):
    tree = KDTree(X, leaf_size=1, metric='chebyshev')
    nnnidx = tree.query_radius(X, r=rV, count_only=True)
    return nnnidx

def nneighforgivenr(X, rV):
    npV = ANNR(X, rV)
    npV[npV == 0] = 1
    return npV

def get_tree(x, metric='chebyshev'):
  from sklearn.neighbors import KDTree
  return KDTree(x, leaf_size=1, metric=metric)

def get_neighbors(tree, x, k=5):
  dists, idx = tree.query(x, k=k + 1)
  return dists, idx

def entropy_estimator_ksg(xV, nnei=5, normalize=False):
  '''
  calculates H(X;Y) using KSG algorithm (with max-norms squares)
  H(X) = psi(N) - psi(k) + k*<log(2*maxdist)>
  '''
  from scipy.special import psi

  n, k = xV.shape
  psi_nnei = psi(nnei)
  psi_n = psi(n)

  if normalize:
      xV = (xV - np.min(xV)) / np.ptp(xV)

  _, distsM = ANN(xV, nnei + 1)
  maxdistV = distsM[:, -1]
  #       H(X) = psi(N) - psi(k) + k*<log(2*maxdist)>
  h = -psi_nnei + psi_n + np.mean(np.log(2*maxdistV))*k
  return h


def find_n_on_marginal(marginal, maxdist):
    marginal_n = np.empty(shape=(marginal.shape[0], 1), dtype=np.int64)
    N, dim = marginal.shape
    temp_maxdistV = maxdist.reshape(-1, 1) - np.ones((N, 1)) * 10 ** (-10)
    for i in np.arange(N):
    # for i in prange(N):
        # counter = np.sum(np.abs(marginal[i] - marginal) < temp_maxdistV[i])
        temp_ = np.abs(marginal[i] - marginal) < temp_maxdistV[i]
        # true_neigh = np.zeros(shape=(N,1), dtype=np.int64)
        # for j in range(N):
        #     for d in range(dim):
        #         true_neigh[j] += temp_[j][d]
        # counter = np.sum(true_neigh == dim)
        counter = np.sum(np.sum(temp_, axis=1) == dim)
        marginal_n[i] = counter
    # if np.sum(marginal_n == 0) > 0:
    #     print(1)
    marginal_n = np.where(marginal_n == 0, 1, marginal_n)
    return marginal_n.reshape(1,-1)[0]

def TE(xV, yV, mx, my, taux=1, tauy=1, nnei=5, T=1, normalize=False):

    # TEnnei estimates the transfer entropy (TE) using nearest neighbors
    # (Kraskov's method). TE Indicates the influence of time series xV on time
    # series yV and vice versa, given to the output.
    # INPUTS
    # xV   : time series 1
    # yV   : time series 2
    # nnei : number of nearest neighbors for density estimation
    # T    : T steps ahead, note that if T>1 the whole future vector of
    #           length T is considered.
    #  mx   : embedding dimension for xV
    # taux : lag for xV
    # my   : embedding dimension for yV
    #  tauy : lag for yV
    # OUTPUTS
    # TExy : transfer entropy from xV to yV

    N = len(xV)
    # #normalize data to [0 ,1]
    def normalizer(data):
        return (data - np.min(data, axis=0)) / np.ptp(data, axis=0)
    # #ψ(k)
    psinnei = psi(nnei)

    # #max embedding dim
    M = np.max([(mx - 1) * taux, (my - 1) * tauy])

    # embedded X
    xM = np.full(shape=(N - M - T, mx), fill_value=np.nan)
    for i_mx in np.arange(mx):
        xM[:, i_mx] = (xV[M - i_mx * taux : N - T - i_mx * taux]).reshape(1, -1)

    # embedded Y
    yM = np.full(shape=(N - M - T, my), fill_value=np.nan)
    for i_my in np.arange(my):
        yM[:, i_my] = (yV[M - i_my * tauy: N - T - i_my * tauy]).reshape(1, -1)

    # #future values
    xpreM = np.full(shape=(N - M - T, T), fill_value=np.nan)
    ypreM = np.full(shape=(N - M - T, T), fill_value=np.nan)

    # #get future response
    for i_T in np.arange(T):
        xpreM[:, i_T] = (xV[M + i_T + 1: N - T + i_T + 1]).reshape(1, -1)
        ypreM[:, i_T] = (yV[M + i_T + 1: N - T + i_T + 1]).reshape(1, -1)

    # #calculate entropies
    #####     xMb = [xM yM zM]; cmikra1n(ypreM, xM, yM, nnei)
    # #scale data to [0 ,1]
    if normalize:
        ypreM = normalizer(ypreM)
        xM = normalizer(xM)
        yM = normalizer(yM)

    # # TExy = I(Y;X_past|Y_past) = I(Y;(X_past, Y_past)) - I(Y;Y_past)
    # #I(Y;(X_past, Y_past))
    mi_yT_xy = mi_estimator_ksg1(xV=ypreM, yV=np.concatenate([xM, yM], axis=1))[0]
    # #I(Y;Y_past) calculated through projection
    mi_yT_y = mi_estimator_from_projection(xV=ypreM, yV=yM, zM=np.concatenate([ypreM, xM, yM], axis=1))
    te_xy = mi_yT_xy - mi_yT_y
    return te_xy

    # xall_M = np.concatenate((ypreM, xM, yM), axis=1)
    # _, distsM = ANN(xall_M, nnei + 1)
    # maxdistV = distsM[:, -1]

    # n_z = nneighforgivenr(X=yM, rV=maxdistV - np.ones(N - M - T) * 10 ** (-10))
    # n_yz = nneighforgivenr(X=np.concatenate((xM, yM), axis=1), rV=maxdistV - np.ones(N - M - T) * 10 ** (-10))
    # n_xz = nneighforgivenr(X=np.concatenate((ypreM, yM), axis=1), rV=maxdistV - np.ones(N - M - T) * 10 ** (-10))
    #
    # psin1 = psi(N - M - T)
    # psi_mar = np.full((N - M - T, 3), np.nan)
    # psi_mar[:, 0] = psi(n_xz)
    # psi_mar[:, 1] = psi(n_yz)
    # psi_mar[:, 2] = -psi(n_z)
    #
    # # I(X;Y) = ψ(k) + ψ(Ν) - <ψ(Nx + 1) + ψ(Ny + 1)>,
    # # I(X;Y|Z) = ψ(k) - <ψ(Nxz + 1) + ψ(Nyz + 1) - ψ(Nz + 1)>,
    # # I(X;Y|Z) = I(X;(Y,Z)) - I(X;Z)
    # # I(X;(Y,Z)) = ψ(k) + ψ(Ν) - <ψ(Nx + 1) + ψ(Nyz + 1)>,
    # # Ip(X;Z) = ψ(Ν) - <ψ(Nx + 1) + ψ(Nz + 1) - ψ(Nxz + 1) >
    # # TExy = I(Y;X_past|Y_past) = I(Y;(X_past, Y_past)) - I(Y;Y_past)
    # # TExy = cmikra1n(ypreM, xM, yM, nnei)
    # # TEyx = cmikra1n(xpreM, yM, xM, nnei)
    # mi = psinnei - np.mean(np.sum(psi_mar, axis=1))
    # '''
    # X = ypreM
    # Y = xV
    # Z = yV
    # '''
    # # n_x = find_n_on_marginal(marginal=ypreM, maxdist=maxdistV)
    # n_x = nneighforgivenr(X=ypreM, rV=maxdistV - np.ones(N - M - T) * 10 ** (-10))
    # psiff = psi(np.concatenate([n_x.reshape(-1, 1), n_yz.reshape(-1, 1)], axis=1))
    #
    # mix_yz = psinnei + psin1 - np.mean(np.sum(psiff, axis=1))
    # mix_z = psin1 - np.mean(np.sum(np.concatenate(
    #                                                 [-psi(n_xz).reshape(-1, 1),
    #                                                psi(n_x).reshape(-1, 1),
    #                                                psi(n_z).reshape(-1, 1)], axis=1
    #                                               ), axis=1))
    #
    # return mix_yz - mix_z # # X --> Y

def cmi_estimator_variableKnn(xV, yV, zM, start_nnei=1, normalize=False, knn_threshold=0.2, upper_k_bound=10):
  '''
  calculates I(X;Y | Z)
  # I(X;Y|Z) = ψ(k) - <ψ(Nxz + 1) + ψ(Nyz + 1) - ψ(Nz + 1)>,
  # I(X;Y|Z) = I(X;(Y,Z)) - I(X;Z)
  # I(X;(Y,Z)) = ψ(k) + ψ(Ν) - <ψ(Nx + 1) + ψ(Nyz + 1)>,
  # Ip(X;Z) = ψ(Ν) - <ψ(Nx + 1) + ψ(Nz + 1) - ψ(Nxz + 1) >

  '''
  from scipy.special import psi
  from ANN import ANN, find_n_on_marginal

  xall_M = np.concatenate((xV, yV, zM), axis=1)
  nnei = int(start_nnei)
  while True:
    _, distsM = ANN(xall_M, nnei + 1)
    maxdistV = distsM[:, -1]
    if (np.mean(maxdistV) < knn_threshold) & (nnei < upper_k_bound + 1):
        valid_nnei = int(nnei)
        valid_distance = np.array(maxdistV)
        nnei += 1
    else:
        if nnei == 1:
            return np.nan, nnei, np.mean(maxdistV)
        else:
            break
  n = xV.shape[0]
  psi_n = psi(n)
  psi_nnei = psi(valid_nnei)

  n_z = nneighforgivenr(X=zM, rV=valid_distance - np.ones(n) * 10 ** (-10))
  n_yz = nneighforgivenr(X=np.concatenate((yV, zM), axis=1), rV=valid_distance - np.ones(n) * 10 ** (-10))
  n_xz = nneighforgivenr(X=np.concatenate((xV, zM), axis=1), rV=valid_distance - np.ones(n) * 10 ** (-10))

  psi_mar = np.full((n, 3), np.nan)
  psi_mar[:, 0] = psi(n_xz)
  psi_mar[:, 1] = psi(n_yz)
  psi_mar[:, 2] = -psi(n_z)

  n_x = find_n_on_marginal(marginal=xV, maxdist=valid_distance)
  psiff = psi(np.concatenate([n_x.reshape(-1, 1), n_yz.reshape(-1, 1)], axis=1))

  mix_yz = psi_nnei + psi_n - np.mean(np.sum(psiff, axis=1))
  mix_z = psi_n - np.mean(np.sum(np.concatenate(
      [-psi(n_xz).reshape(-1, 1),
       psi(n_x).reshape(-1, 1),
       psi(n_z).reshape(-1, 1)], axis=1
  ), axis=1))

  # cmi_xy_z = psi_nnei - np.mean(np.sum(psi_mar, axis=1))
  cmi_xy_z = mix_yz - mix_z
  return cmi_xy_z, valid_nnei, np.mean(valid_distance)

def mi_estimator_ksg_variableKnn(xV, yV, start_nnei=1, normalize=False, knn_threshold=0.2, upper_k_bound=10):
  '''
  calculates I(X;Y) using KSG algorithm1 (with max-norms squares)
  '''
  from scipy.special import psi

  if normalize:
      xV = (xV - np.min(xV)) / np.ptp(xV)
      yV = (yV - np.min(yV)) / np.ptp(yV)

  xembM = np.concatenate((xV, yV), axis=1)
  nnei = int(start_nnei)
  while True:
    _, distsM = ANN(xembM, nnei + 1)
    maxdistV = distsM[:, -1]
    if (np.mean(maxdistV) < knn_threshold) & (nnei < upper_k_bound + 1):
        valid_nnei = int(nnei)
        valid_distance = np.array(maxdistV)
        nnei += 1
    else:
        if nnei == 1:
            return np.nan, nnei, np.mean(maxdistV)
        else:
            break
  n = xembM.shape[0]
  n_x = nneighforgivenr(X=xV, rV=valid_distance - np.ones(n) * 10 ** (-10))
  n_y = nneighforgivenr(X=yV, rV=valid_distance - np.ones(n) * 10 ** (-10))
  psibothM = psi(np.concatenate((n_x.reshape(-1, 1), n_y.reshape(-1, 1)), axis=1))
  psi_nnei = psi(valid_nnei)
  psi_n = psi(n)
  #     # I(X;Y) = ψ(k) + ψ(Ν) - <ψ(Nx + 1) + ψ(Ny + 1)>
  mi = psi_nnei + psi_n - np.mean(np.sum(psibothM, axis=1))
  return mi, valid_nnei, np.mean(valid_distance)

def gen_a_var(p, d, max_eig, edge_density, nonzero_diag):
    '''
    :param p: number of variables
    :param d: lag of the model
    :param max_eig: spectral radius of A
    :param edge_density: if different for different lags,
    provide a vector of length d
    :param nonzero_diag: ensure all diagonal entries are non-zero, if different for different lags,
    provide a d-vector 0 and 1
    :return: matrix A (p x p x d), p x p transition matrix for each lag
    '''
    a = np.full(shape=(p, p, d), fill_value=np.nan)
    if len(edge_density) == 1:
        edge_density = np.tile(edge_density, (1, d))
    if len(nonzero_diag) == 1:
        nonzero_diag = np.tile(nonzero_diag, (1, d))

    for lag in np.arange(d):
        ed = int(np.ceil((p**2) * edge_density[lag]))
        temp = np.random.randint(p**2 , size=ed).reshape(-1, 1)
        temp_v = np.zeros(shape=(p**2, 1))
        temp_v[temp] = 1
        a[:, :, lag] = temp_v.reshape(p, p)
        if nonzero_diag[lag] == 1:
            for lag1 in np.arange(p):
                a[lag1, lag1, lag] = 1

    a_big = np.zeros(shape=(p * d, p * d))
    for i in np.arange(d):
        a_big[:p, i * p : ((i + 1) * p)] = max_eig * a[:, :, i]
    if d > 1:
        s1 = a_big[p:, : (d - 1) * p].shape
        a_big[p: , : (d - 1) * p] = np.eye(s1[0], s1[1])
        temp = np.max((np.abs(np.linalg.eig(a_big)[0])))
        count = 0
        while temp > max_eig:
            count += 1
            print(f'count:{count} max_eigen_value:{round(temp * 100) / 100}'
                  f' signal:{round((abs(a_big[1:p,:]).max()*100).max()) / 100}'
                   )

            a_big[:p, :] *= 0.95 # decrease signal strength till the model gets stationary
            temp = np.max((np.abs(np.linalg.eig(a_big)[0])))
    for i in np.arange(d):
        a[:, :, i] = a_big[:p, i * p : ((i + 1) * p)]
    return a

def generate_a_var_given_coupling(p, d, max_eig, coupling_matrix, nonzero_diag):
    '''
    :param p: number of variables
    :param d: lag of the model
    :param max_eig: spectral radius of A
    :param edge_density: if different for different lags,
    provide a vector of length d
    :param nonzero_diag: ensure all diagonal entries are non-zero, if different for different lags,
    provide a d-vector 0 and 1
    :return: matrix A (p x p x d), p x p transition matrix for each lag
    '''
    a = np.full(shape=(p, p, d), fill_value=0.)
    # if len(edge_density) == 1:
    #     edge_density = numpy.matlib.repmat(edge_density, 1, d)
    if len(nonzero_diag) == 1:
        nonzero_diag = np.tile(nonzero_diag, (1, d))

    ####get connections indices
    connections_pos = np.argwhere(coupling_matrix > 0)
    for i in connections_pos:
        random_lag = np.random.randint(d)
        conn_position_on_a = np.append(i[:], random_lag)
        a[tuple(conn_position_on_a)] = 1.
    for lag in np.arange(d):
        if nonzero_diag[lag] == 1:
            for lag1 in np.arange(p):
                a[lag1, lag1, lag] = 1

    a_big = np.zeros(shape=(p * d, p * d))
    for i in np.arange(d):
        a_big[:p, i * p: ((i + 1) * p)] = max_eig * a[:, :, i]
    if d > 1:
        s1 = a_big[p:, : (d - 1) * p].shape
        a_big[p:, : (d - 1) * p] = np.eye(s1[0], s1[1])
        temp = np.max((np.abs(np.linalg.eig(a_big)[0])))
        count = 0
        while temp > max_eig:
            count += 1
            print(f'count:{count} max_eigen_value:{round(temp * 100) / 100}'
                  f' signal:{round((abs(a_big[1:p, :]).max() * 100).max()) / 100}'
                  )

            a_big[:p, :] *= 0.95  # decrease signal strength till the model gets stationary
            temp = np.max((np.abs(np.linalg.eig(a_big)[0])))
    for i in np.arange(d):
        a[:, :, i] = a_big[:p, i * p: ((i + 1) * p)]
    return a

def get_crosscorr(xV, yV, tau):
    if tau == 0:
        return np.corrcoef(xV, yV)[0, 1]
    else:
        ccV = np.full(shape=(2*tau + 1, 1), fill_value=np.nan)
        for i_tau in np.arange(1, tau+1):
            temp_ = np.corrcoef(yV[:-i_tau], xV[i_tau:])
            ccV[i_tau-1] = temp_[0, 1]
        temp_ = np.corrcoef(xV, yV)
        ccV[i_tau] = temp_[0, 1]
        for i_tau in np.arange(1, tau+1):
            temp_ = np.corrcoef(xV[:-i_tau], yV[i_tau:])
            ccV[i_tau + tau] = temp_[0, 1]
    return ccV

def gen_a_var_with_coef_noise(p, d, max_eig, edge_density, nonzero_diag, coef_noise=0.1):
    '''
    :param p: number of variables
    :param d: lag of the model
    :param max_eig: spectral radius of A
    :param edge_density: if different for different lags,
    provide a vector of length d
    :param nonzero_diag: ensure all diagonal entries are non-zero, if different for different lags,
    provide a d-vector 0 and 1
    coef_noise: given as percentage of coefficients
    :return: matrix A (p x p x d), p x p transition matrix for each lag with unequal coefs

    '''
    a = np.full(shape=(p, p, d), fill_value=np.nan)
    if len(edge_density) == 1:
        edge_density = np.tile(edge_density, (1, d))
    if len(nonzero_diag) == 1:
        nonzero_diag = np.tile(nonzero_diag, (1, d))

    for lag in np.arange(d):
        ed = int(np.ceil((p**2) * edge_density[lag]))
        temp = np.random.randint(p**2 , size=ed).reshape(-1, 1)
        temp_v = np.zeros(shape=(p**2, 1))
        temp_v[temp] = 1
        a[:, :, lag] = temp_v.reshape(p, p)
        if nonzero_diag[lag] == 1:
            for lag1 in np.arange(p):
                a[lag1, lag1, lag] = 1

    a_big = np.zeros(shape=(p * d, p * d))
    for i in np.arange(d):
        a_big[:p, i * p : ((i + 1) * p)] = max_eig * a[:, :, i]
    if d > 1:
        s1 = a_big[p:, : (d - 1) * p].shape
        a_big[p: , : (d - 1) * p] = np.eye(s1[0], s1[1])
        temp = np.max((np.abs(np.linalg.eig(a_big)[0])))
        count = 0
        while temp > max_eig:
            count += 1
            print(f'count:{count} max_eigen_value:{round(temp * 100) / 100}'
                  f' signal:{round((abs(a_big[1:p,:]).max()*100).max()) / 100}'
                   )

            a_big[:p, :] *= 0.95 # decrease signal strength till the model gets stationary
            temp = np.max((np.abs(np.linalg.eig(a_big)[0])))

    for i in np.arange(d):
        a[:, :, i] = a_big[:p, i * p: ((i + 1) * p)]
        noise = np.random.normal(0, coef_noise, (p, p)) + 1
        a[:, :, i] = a[:, :, i] * noise
    temp_a = a.transpose(0, 2, 1).reshape(p, -1)
    np.linalg.eig(temp_a)

    return a

def get_timeshift_given_autocorr(xM, maxtau=20, alpha=0.05):
    from statsmodels.tsa.stattools import acf
    n, d = xM.shape
    threshold = 2 / np.sqrt(n)
    maxacf_signV = np.full(shape=d, fill_value=np.nan)
    for i in np.arange(d):
        acf_ = acf(xM[:, [i]], nlags=maxtau)
        filter_sign_acf = (np.abs(acf_) > threshold)
        signif_lags = np.sum(filter_sign_acf)
        maxacf_signV[i] = signif_lags
    return maxacf_signV

def pink_noise(n=2048, ind=1, offset=0):
    '''
    generates time series with a power spectrum of 1/f^ind where
    f is the frequency and 'ind' is the spectral exponent. An offset from the
    origin of frequencies can be given, so that the slope starts beyond the
    given 'offset'.
    :param n: length of the time series to be generated
    :param ind: spectral exponent
    :param offset: offset for the start of the spectral slope
    :return:  generated time series
    '''
    n2 = int(np.floor(n/2))
    fV = np.linspace(0, 0.5, n2) # the frequencies
    fV = fV[1:].reshape(-1, 1)
    ampV = fV ** (-ind/2) #the spectral exponent is halved because it regards the amplitudes of FT, i.e. square root of power spectrum.
    ifV = (fV < offset)
    if np.sum(ifV) > 0:
        ampV[ifV] = ampV[ifV][-1]
    angV = (np.random.rand(n2-1, 1) -0.5) * 2 * np.pi
    yft1V = ampV * np.exp(1j*angV)
    yftV = np.vstack([1, yft1V, (n2+1)**(-ind/2), np.flipud(np.conj(yft1V))])
    yV = np.fft.ifft(yftV.reshape(1, -1))
    return np.real_if_close(yV).reshape(-1, 1)

def calculate_network_statistics(rm):
    '''
    calculates multiple network statistics given an
    adjacency matrix (directed or not)
    :param rm:
    :return: in_degrees, out_degrees, indegree_centrality, outdegree_centrality, betweenness_centrality, closeness_centrality, hubs-authorities(HITS algorithm)
    dicts where key corresponds to node i,
    '''
    rm[rm > 0] = 1
    rm[rm != 1] = 0
    causal_graph = create_causality_network(connectivity_matrix=rm)
    indegree_centrality = nx.in_degree_centrality(causal_graph)
    outdegree_centrality = nx.out_degree_centrality(causal_graph)
    betweenness_centrality = nx.betweenness_centrality(causal_graph)
    closeness_centrality = nx.closeness_centrality(causal_graph)
    in_degrees_ = causal_graph.in_degree
    out_degrees_ = causal_graph.out_degree
    try:
        hubs, authorities = nx.algorithms.hits(causal_graph)
    except Exception as err:
        print(err)
        hubs, authorities = {}, {}
    in_degrees = {node:indeg for node, indeg in in_degrees_}
    out_degrees = {node:outdeg for node, outdeg in out_degrees_}
    return in_degrees, out_degrees, indegree_centrality, outdegree_centrality, betweenness_centrality, closeness_centrality, hubs, authorities

def generate_VARsurrogates(xV, nsur=100, Lmax=3, show=False):
    from statsmodels.tsa.api import VAR
    n, k = xV.shape
    # #gaussianize initial variables
    FxV = (np.arange(1, n + 1) - 0.326) / (n + 0.348)
    owV = stats.norm.ppf(FxV)
    yM = np.full(shape=(n, k), fill_value=np.nan)
    oxM = np.full(shape=(n, k), fill_value=np.nan)
    for j in np.arange(k):
        idx = np.argsort(xV[:, j])
        oxM[:, j] = xV[idx, j]
        ixV = np.argsort(idx)
        yM[:, j] = owV[ixV]
    # #check aic - bic for p={1,...,Lmax}
    # info_metrics = dict(aic=[], bic=[])
    # for p in np.arange(1, Lmax+1):
    #     var = VAR(yM).fit(maxlags=p)
    #     aic_ = var.aic
    #     bic_ = var.bic
    #     info_metrics['aic'].append(aic_)
    #     info_metrics['bic'].append(bic_)
    # #fit VAR(p=Lmax)
    var = VAR(yM).fit(maxlags=Lmax)
    # #generate data from VAR - nsur times
    zsurT = np.full(shape=(n, k, nsur), fill_value=np.nan)
    for isur in np.arange(nsur):
        ysurM = var.simulate_var(steps=n)
        zsurM = np.full(shape=(n, k), fill_value=np.nan)
        # #invert gaussianisation for each variable
        for j in np.arange(k):
            idx = np.argsort(ysurM[:, j])
            isurV = np.argsort(idx)
            zsurM[:, j] = oxM[isurV, j]
        zsurT[:, :, isur] = zsurM
    if show:
        for i in np.arange(5):
            temp = zsurT[:, :, i]
            fig, ax = plt.subplots(temp.shape[1], 1)
            for j in range(temp.shape[1]):
                ax[j].plot(temp[:, j])
                ax[j].set_yticks([])
        plt.show()
    return zsurT#, info_metrics

def generate_VARsurrogates_withAICBIC(xV, nsur=100, Lmax=3, show=False):
    from statsmodels.tsa.api import VAR
    n, k = xV.shape
    # #gaussianize initial variables
    FxV = (np.arange(1, n + 1) - 0.326) / (n + 0.348)
    owV = stats.norm.ppf(FxV)
    yM = np.full(shape=(n, k), fill_value=np.nan)
    oxM = np.full(shape=(n, k), fill_value=np.nan)
    for j in np.arange(k):
        idx = np.argsort(xV[:, j])
        oxM[:, j] = xV[idx, j]
        ixV = np.argsort(idx)
        yM[:, j] = owV[ixV]
    # #check aic - bic for p={1,...,Lmax}
    info_metrics = dict(aic=[], bic=[])
    p_aic = np.inf
    p_best = None
    for p in np.arange(0, Lmax+1):
        var = VAR(yM).fit(maxlags=p)
        aic_ = var.aic
        bic_ = var.bic
        if aic_ < p_aic:
            p_best = p
            p_aic = aic_
        info_metrics['aic'].append(aic_)
        info_metrics['bic'].append(bic_)
    # #fit VAR(p=Lmax)
    var = VAR(yM).fit(maxlags=p_best)
    # #generate data from VAR - nsur times
    zsurT = np.full(shape=(n, k, nsur), fill_value=np.nan)
    for isur in np.arange(nsur):
        ysurM = var.simulate_var(steps=n)
        zsurM = np.full(shape=(n, k), fill_value=np.nan)
        # #invert gaussianisation for each variable
        for j in np.arange(k):
            idx = np.argsort(ysurM[:, j])
            isurV = np.argsort(idx)
            zsurM[:, j] = oxM[isurV, j]
        zsurT[:, :, isur] = zsurM
    if show:
        for i in np.arange(5):
            temp = zsurT[:, :, i]
            fig, ax = plt.subplots(temp.shape[1], 1)
            for j in range(temp.shape[1]):
                ax[j].plot(temp[:, j])
                ax[j].set_yticks([])
        plt.show()
    return zsurT, info_metrics

def generate_var1_with_communities(k, phi=0.9, h=3, hidden_strength=0.8, hidden_acf=0.3, n=1000, clusters=None, global_acf=None, global_strength=None):
    '''
    generate VAR(1) system organized in communities drived by hidden sources
    :param k: number of variables
    :param phi: acf of each of variable
    :param h: number of hidden sources
    :param hidden_strength: effect of each of the hidden source on each variable
    :param hidden_acf: acf of each of the hidden source
    :param n: length of time series
    :param clusters: list which elements are the user-defined clusters,
    if None, generate equal-sized clusters
    :return: xM
    '''

    if hidden_strength == 0:
        h = int(k)
    if clusters is None:
        clusters = []
        cluster_size = k//h
        startcluster_id = 0
        endcluster_id = cluster_size

        for _ in np.arange(h):
            cluster = np.arange(startcluster_id, endcluster_id)
            startcluster_id += cluster_size
            endcluster_id += cluster_size
            clusters.append(tuple(cluster))
        # clusters = [(np.arange(15)), (np.arange(15, 30)),(np.arange(30,k))]

    # set A matrix
    aM = np.eye(k+h) * phi
    for i in np.arange(h):
        aM[k+i, k+i] = hidden_acf

    for icluster in np.arange(len(clusters)):
        for inode in clusters[icluster]:
            aM[inode, k+icluster] = hidden_strength

    if global_strength is not None:
        global_effect = np.zeros(shape=(k+h,1))
        global_effect[:-h] = global_strength
        aM = np.concatenate([aM, global_effect], axis=1)
        global_acfV = np.zeros(shape=(1, k+h+1))
        global_acfV[0, -1] = global_acf
        aM = np.concatenate([aM, global_acfV], axis=0)

    wM = np.random.normal(size=(n, aM.shape[1]))
    xM = np.full(shape=(n, aM.shape[1]), fill_value=np.nan)
    xM[0, :] = wM[0, :]
    for i in np.arange(1, n):
        xM[i, :] = np.dot(aM, xM[i-1, :]) + wM[i]
    xM = xM[:, :k]
    # if clusters == None:
    true_partition = []
    partitions = aM[:k, k:]
    for i in np.arange(h):
        current_cluster = np.argwhere(partitions[:,i]).reshape(-1,)
        if len(current_cluster) == 0:
            current_cluster = [i]
        true_partition.append(tuple(current_cluster))
    return xM, true_partition

def vi_discrete(X, Y):
    '''
    calculates Variation of Information in the discrete case
    VI(X;Y) = 1 - I(X;Y) / H(X,Y)
    :returns [0,1], 0--> identical, 1--> max difference
    '''
    def entropy(X):
        probs = [np.mean(X == c) for c in set(X)]
        return np.sum(-p * np.log2(p) for p in probs)
    def joint_entropy(X,Y):
        probs = []
        for c1 in set(X):
            for c2 in set(Y):
                probs.append(np.mean(np.logical_and(X == c1, Y == c2)))

        return np.sum(-p * np.log2(p) for p in probs if p !=0)
    h_x = entropy(X)
    h_y = entropy(Y)
    h_xy = joint_entropy(X, Y)
    if h_xy == 0:
        return 1
    return 1 - (h_x + h_y - h_xy) / h_xy

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=5.)

    pos_nodes = _position_nodes(g, partition, scale=2.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def adjFDRmatrix(pM, alpha=0.05, symmetric=False):
    '''
    % adjM = adjFDRmatrix(xM,alpha,symmetric)
    % Function adjFDRmatrix creates an adjacency matrix of zeros and ones from
    % a given square KxK matrix 'xM' of p-values, where ones regards
    % significant p-values according to the False Discovery Rate (FDR)
    % criterion for a given 'alpha'. If symmetric~=0, then 'xM' is supposed to
    % be symmetric and only the upper (or lower) triangular off diagonal matrix
    % is considered (K*(K-1)/2 values), otherwise all but the diagonal
    % components are considered (K*(K-1) values).
    % INPUT
    % - xM          : square K x K matrix of p-values.
    % - alpha       : the significance limit for FDR
    % - symmetric   : if ~=0, xM is to be considered as symmetric (default ==0)
    % OUTPUT
    % - adjM        : the adjacency K x K matrix zeros and ones (ones for
    %                 significant rejection, small p-values).
    '''
    K,K1 = pM.shape
    assert K == K1, 'The input matrix of p-values must be square.'


    if symmetric:
        m = int(K*(K-1)/2)
        xvecM = np.full(shape=(m,3), fill_value=np.nan)
        count = 0
        for i in np.arange(K):
            jV = (np.arange(i+1, K)).reshape(-1,1)
            nj = len(jV)
            xvecM[(np.arange(nj) + count).reshape(-1,1), 0] = np.ones(shape=(nj, 1))*i
            xvecM[(np.arange(nj) + count).reshape(-1,1), 1] = jV
            xvecM[(np.arange(nj) + count).reshape(-1,1), 2] = pM[i, jV]
            count = count+nj
    else:
        m = int(K*(K-1))
        xvecM = np.full(shape=(m,3), fill_value=np.nan)
        count = 0
        for i in np.arange(K):
            jV = np.setdiff1d(np.arange(K), i).reshape(-1,1)
            nj = len(jV)
            xvecM[(np.arange(nj) + count).reshape(-1,1), 0] = np.ones(shape=(nj, 1))*i
            xvecM[(np.arange(nj) + count).reshape(-1,1), 1] = jV
            xvecM[(np.arange(nj) + count).reshape(-1,1), 2] = pM[i, jV]
            count = count+nj

    ixvecV = np.argsort(xvecM[:,2])
    oxvecV = xvecM[ixvecV, 2]
    iV = np.where(oxvecV <= alpha*np.arange(1, m+1)/m)[0]
    adjM = np.zeros(shape = (K, K))

    if len(iV) > 0:
        ixvecV = ixvecV[0:iV[-1]+1]
        for i in np.arange(len(ixvecV)):
            adjM[int(xvecM[ixvecV[i],0]), int(xvecM[ixvecV[i],1])] = 1
    if symmetric:
        adjM[int(xvecM[ixvecV[i],1]), int(xvecM[ixvecV[i],0])] = 1
    return adjM
