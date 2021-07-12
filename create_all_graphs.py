import matplotlib.pyplot as plt
import numpy as np
import os, re
from sys import path as syspath

syspath.append('C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\OOP_hard_sphere_event_chain')
from post_process import Ising, Graph
from SnapShot import WriteOrLoad
from mpl_toolkits import mplot3d
import scipy
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import *

sims_dir = 'C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\post_process\\from_ATLAS3.0'
default_plt_kwargs = {'linewidth': 5, 'markersize': 10}
size = 30
params = {'legend.fontsize': size * 0.75, 'figure.figsize': (10, 10), 'axes.labelsize': size, 'axes.titlesize': size,
          'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75}
plt.rcParams.update(params)
colors_rho = {0.85: 'C0', 0.83: 'C1', 0.8: 'C2', 0.78: 'C3', 0.77: 'C4', 0.785: 'C5', 0.775: 'C6',
              0.75: 'C7'}


def join(*args, **kwargs):
    return os.path.join(*args, **kwargs)


def sim_name(rhoH, N=9e4, h=0.8, initial_conditions='AF_square'):
    return 'N=' + str(int(N)) + '_h=' + str(h) + '_rhoH=' + str(rhoH) + '_' + initial_conditions + '_ECMC'


def op_path(rhoH, specif_op=None, *args, **kwargs):
    op_dir = join(sims_dir, sim_name(rhoH, *args, **kwargs), 'OP')
    if specif_op is None:
        return op_dir
    else:
        return join(op_dir, specif_op)


def sort_prefix(folder_path, prefix='correlation_', surfix='.txt', reverse=True):
    relevent_files = [file for file in os.listdir(folder_path) if file.startswith(prefix) and file.endswith(surfix)]
    reals = [int(re.split('\.', re.split('_', file)[1])[0]) for file in relevent_files]
    sorted_files = [f for _, f in sorted(zip(reals, relevent_files), reverse=reverse)]
    sorted_reals = sorted(reals, reverse=reverse)
    return sorted_files, sorted_reals


def prepare_lbl(lbl):
    lbl = re.sub('_', ' ', lbl)
    lbl = re.sub('rhoH', '$\\\\rho_H$', lbl)
    if lbl.startswith('psi'):
        for mn in ['14', '23', '16', '4', '6']:
            lbl = re.sub('psi ' + mn, '$g_{' + mn + '}$', lbl)
    if lbl.startswith('Bragg Sm'):
        lbl = re.sub('Bragg Sm', '$g_k^M$', lbl)
    if lbl.startswith('Bragg S'):
        lbl = re.sub('Bragg S', '$g_k$', lbl)
    for N, N_ in zip(['10000', '40000', '90000'], ['1e4', '4e4', '9e4']):
        lbl = re.sub(N, N_, lbl)
    return lbl


def plot_corr(rhoH, specific_op, realizations=1, poly_slope=None, single_slope_label=False, *args,
              **kwargs):
    pol = (poly_slope is not None)
    if pol:
        maxys = []
        maxxs = []
        slopes = []
    if type(rhoH) is not list:
        rhoH = [rhoH]
    for rho in rhoH:
        op_dir = op_path(rho, specific_op, **kwargs)
        xs, ys = [], []
        for vec_file in sort_prefix(op_dir)[0][:realizations]:
            x, y = np.loadtxt(join(op_dir, vec_file), unpack=True, usecols=(0, 1))
            x = np.array(x) / 2.0
            xs.append(x)
            ys.append(y)
            if pol:
                I = np.where(np.logical_and(x > 0.5, x < 1.5))
                maxys.append(np.nanmean(y[I]))
                maxxs.append(2)
                cond = lambda x, y: x > 10 and x < 20 and (not np.isnan(y))
                y_p = np.array([y_ for x_, y_ in zip(x, y) if cond(x_, y_)])
                x_p = np.array([x_ for x_, y_ in zip(x, y) if cond(x_, y_)])
                p = np.polyfit(np.log(x_p), np.log(y_p), 1)
                slopes.append(-p[0])
        x = xs[np.argmin([len(x) for x in xs])]
        y = np.mean(ys[:len(x)], 0)
        plt.loglog(x, y, colors_rho[rho], label=prepare_lbl('rhoH=' + str(rho)), **default_plt_kwargs)
    if pol:
        I = np.argsort(slopes)
        maxys = np.array(maxys)[I]
        maxxs = np.array(maxxs)[I]
        slopes = np.array(slopes)[I]
        if min(slopes) > poly_slope:
            y_init = min(maxys)
            x_init = maxxs[np.argmin(maxys)]
        else:
            if max(slopes) < poly_slope:
                y_init = max(maxys)
                x_init = maxxs[np.argmax(maxys)]
            else:
                i = np.where(slopes > poly_slope)[0][0]
                y_init = maxys[i - 1]
                x_init = maxxs[i - 1]
        y = y_init * np.power(x / x_init, -poly_slope)
        lbl = '$x^{-1/4\ or -1/3}$' if not single_slope_label else '$x^{-' + str(np.round(poly_slope, 2)) + '}$'
        plt.loglog(x, y, '--k', label=lbl, **default_plt_kwargs)
    plt.grid()
    plt.legend(loc=4)
    return


def plot_pos_and_orientation(rhos_pos, rhos_psi):
    plt.rcParams.update({'figure.figsize': (10, 10)})
    plt.figure()
    corr_ylim = [1e-2, 1]
    corr_xlim = [0.8, 1e2]

    plt.subplot(211)
    plot_corr(rhos_pos, 'Bragg_S', poly_slope=1.0 / 3)
    plt.ylim(corr_ylim)
    plt.xlim(corr_xlim)
    plt.ylabel(prepare_lbl('Bragg_S'))
    plt.legend([prepare_lbl('rhoH=' + str(r)) for r in rhos_pos] + ['$x^{-1/3}$'], loc=3)
    # ax = plt.gca()
    # ax.legend_ = None

    plt.subplot(212)
    plot_corr(rhos_psi, 'psi_14', poly_slope=0.25, N=4e4)
    plt.ylim(corr_ylim)
    plt.xlim(corr_xlim)
    plt.ylabel(prepare_lbl('psi_4'))
    plt.xlabel('$\Delta$r/$\sigma$')
    plt.legend([prepare_lbl('rhoH=' + str(r)) for r in rhos_psi] + ['$x^{-1/4}$'], loc=3)

    plt.savefig('graphs/orientation_and_position_corr')


def quiver_burger(rhoH, xlim, ylim, realization=None, bonds=False, frustrated_bonds=True, quiv=True,
                  orientational_rad=None, *args, **kwargs):
    plt.rcParams.update({'figure.figsize': (15, 8)})
    plt.figure()
    op_name = 'burger_vectors'
    if orientational_rad is not None:
        op_name += '_orientation_rad=' + str(orientational_rad)
    op_dir = op_path(rhoH, op_name, *args, **kwargs)
    # op_dir = op_path(rhoH, 'burger_vectors_orientation_rad=10.0', *args, **kwargs)
    sim = join(sims_dir, sim_name(rhoH, *args, **kwargs))
    if realization is None:
        files, reals = sort_prefix(op_dir, 'vec_')
        realization = reals[0]
        file = files[0]
    else:
        file = 'vec_' + str(realization) + '.txt'
    spheres = np.loadtxt(join(sim, str(realization)))
    if bonds:
        op = Ising(sim_path=sim, k_nearest_neighbors=4, directed=False)
        op.update_centers(spheres, realization)
        spheres = op.spheres  # might reogranize order
    x = spheres[:, 0] / 2
    y = spheres[:, 1] / 2
    z = spheres[:, 2] / 2
    up = np.array(
        [(z_ > np.mean(z)) and xlim[0] < x_ < xlim[1] and ylim[0] < y_ < ylim[1] for (x_, y_, z_) in zip(x, y, z)])
    down = np.array(
        [(z_ <= np.mean(z)) and xlim[0] < x_ < xlim[1] and ylim[0] < y_ < ylim[1] for (x_, y_, z_) in zip(x, y, z)])
    if bonds:
        op.initialize(random_initialization=False, J=-1)
        spins = op.z_spins
        for i in range(len(x)):
            for j in op.nearest_neighbors[i]:
                if j > i or not (up[i] or down[i]) or not (up[j] or down[j]):
                    continue
                ex = [x[i], x[j]]
                ey = [y[i], y[j]]
                if (ex[1] - ex[0]) ** 2 + (ey[1] - ey[0]) ** 2 > 10 ** 2:
                    continue
                if spins[i] * spins[j] > 0 and frustrated_bonds:
                    plt.plot(ex, ey, color='green', linewidth=3)
                if spins[i] * spins[j] < 0:
                    plt.plot(ex, ey, color='lightgray')
    # plt.plot(x[up], y[up], '.', color='orange', label='up', markersize=10)
    plt.plot(x[up], y[up], '.r', label='up', markersize=10)
    plt.plot(x[down], y[down], '.b', label='down', markersize=10)
    plt.axis('equal')
    # plt.legend()
    if quiv:
        burg = np.loadtxt(join(op_dir, file)) / 2
        I_box = np.array(
            [xlim[0] < x_ < xlim[1] and ylim[0] < y_ < ylim[1] for (x_, y_) in zip(burg[:, 0], burg[:, 1])])
        burg = burg[I_box, :]
        plt.quiver(burg[:, 0], burg[:, 1], burg[:, 2], burg[:, 3], angles='xy', scale_units='xy', scale=1,
                   label='Burger field', width=3e-3, zorder=7)  # headwidth=3)  # , headlength=10, headaxislength=6
    plt.savefig('graphs/burger_vectors')
    return


def cyc_dist(p1, p2, boundaries):
    dx = np.array(p1) - p2  # direct vector
    dsq = 0
    for i in range(2):
        L = boundaries[i]
        dsq += min(dx[i] ** 2, (dx[i] + L) ** 2, (dx[i] - L) ** 2)  # find shorter path through B.D.
    return np.sqrt(dsq)


def clean_bound_pairs(burg, sim_path, cutoff=np.inf):
    # save graph into real num with 000 to differentiate from usual graph saves.
    write_or_load = WriteOrLoad(sim_path)
    l_x, l_y, _, _, _, _, _, _ = write_or_load.load_Input()
    cyc = lambda p1, p2: cyc_dist(p1, p2, [l_x / 2, l_y / 2])
    graph = kneighbors_graph(burg[:, :2], n_neighbors=1, metric=cyc)
    I, J, _ = scipy.sparse.find(graph)[:]
    Ed = [(i, j) for (i, j) in zip(I, J)]
    Eud = []
    N = len(burg)
    udgraph = scipy.sparse.csr_matrix((N, N))
    for i, j in Ed:
        if ((j, i) in Ed) and ((i, j) not in Eud) and ((j, i) not in Eud):
            Eud.append((i, j))
            udgraph[i, j] = 1
            udgraph[j, i] = 1
    graph = udgraph
    nearest_neighbor = [[j for j in graph.getrow(i).indices] for i in range(N)]
    cleaned_burg = []
    for i in range(N):
        r1 = np.array([burg[i, 0], burg[i, 1]])
        b1 = np.array([burg[i, 2], burg[i, 3]])
        if len(nearest_neighbor[i]) == 0:
            cleaned_burg.append([r1[0], r1[1], b1[0], b1[1]])
            continue
        j = nearest_neighbor[i][0]
        if i > j:
            continue
        r2 = [burg[j, 0], burg[j, 1]]
        b2 = [burg[j, 2], burg[j, 3]]
        if cyc(r1, r2) > cutoff or np.linalg.norm(b1 + b2) > 1e-5:
            cleaned_burg.append([r1[0], r1[1], b1[0], b1[1]])
            cleaned_burg.append([r2[0], r2[1], b2[0], b2[1]])
    return np.array(cleaned_burg)


def clean_by_cluster(burg, algorithm, *args, **kwargs):
    model = algorithm(*args, **kwargs)
    try:
        model.fit(burg[:, :2])
        yhat = model.predict(burg[:, :2])
    except AttributeError:
        yhat = model.fit_predict(burg[:, :2])
    clusters = np.unique(yhat)
    clean_burgers = []
    for cluster_ind in clusters:
        cluster = np.where(yhat == cluster_ind)[0]
        if np.linalg.norm(np.sum(burg[cluster, 2:])) > 1e-5:
            for dislocation_ind in cluster:
                clean_burgers.append([burg[dislocation_ind, i] for i in range(4)])
    return np.array(clean_burgers)


def plot_clustering(burgers, algorithm, rhoH=None, *args, **kwargs):
    X = burgers[:, :2]
    model = algorithm(*args, **kwargs)
    try:
        model.fit(X)
        yhat = model.predict(X)
    except AttributeError:
        yhat = model.fit_predict(X)
    clusters = np.unique(yhat)
    cluster_parity = []
    a = min([np.linalg.norm(b) for b in burgers[:, 2:4]])
    for cluster in clusters:
        row_ix = np.where(yhat == cluster)
        sum_b = np.sum(burgers[row_ix, 2:4], 0)
        cluster_parity.append(int(np.linalg.norm(sum_b) ** 2 / a ** 2))
        if cluster_parity[-1] % 2 == 0:
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
        else:
            plt.scatter(X[row_ix, 0], X[row_ix, 1], marker='*')
    return np.array(cluster_parity)


def quiver_cleaned_burgers(rhoH, realization=None, pair_cleans_iterations=1, cutoff=10, xlim=[], ylim=[],
                           overwrite_pair_cleaning=False, clean_by_cluster_flag=False, cluster_algorithm=None,
                           cluster_args={}, overwrite_cluster_clean=True, color_by_cluster=False, *args, **kwargs):
    plt.rcParams.update({'figure.figsize': (15, 8)})
    plt.figure()
    op_dir = op_path(rhoH, 'burger_vectors', *args, **kwargs)
    # op_dir = op_path(rhoH, 'burger_vectors_orientation_rad=10.0', *args, **kwargs)
    sim = join(sims_dir, sim_name(rhoH, *args, **kwargs))
    if realization is None:
        files, reals = sort_prefix(op_dir, 'vec_')
        realization = reals[0]
        file = files[0]
    else:
        file = 'vec_' + str(realization) + '.txt'
    plt.axis('equal')
    # plt.legend()
    burg = np.loadtxt(join(op_dir, file)) / 2
    for i in range(pair_cleans_iterations):
        clean_path = join(op_dir, 'vec_' + str(realization) + '_pair-cleaned-' + str(i + 1) + '-times.txt')
        if os.path.exists(clean_path) and not overwrite_pair_cleaning:
            burg = np.loadtxt(clean_path)
        else:
            burg = clean_bound_pairs(burg, sim, cutoff=cutoff)
            np.savetxt(clean_path, burg)
    if clean_by_cluster_flag:
        clean_name = 'vec_' + str(realization) + '_cluster-cleaned.txt'
        clean_path = join(op_dir, clean_name)
        if os.path.exists(clean_path) and (not overwrite_cluster_clean):
            burg = np.loadtxt(clean_path)
        else:
            burg = clean_by_cluster(burg, cluster_algorithm, **cluster_args)
            np.savetxt(clean_path, burg)
    plt.quiver(burg[:, 0], burg[:, 1], burg[:, 2], burg[:, 3], angles='xy', scale_units='xy', scale=1,
               label='Burger field', width=3e-3, zorder=7)  # headwidth=3)  # , headlength=10, headaxislength=6
    plt.title(('Microscopic Burgers' if pair_cleans_iterations == 0
               else ('Pair Cleaned ' + str(pair_cleans_iterations) + ' times')) + \
              ('' if (not clean_by_cluster_flag) else ', cluster cleaned'))
    if len(xlim) > 0:
        plt.xlim(xlim)
    if len(ylim) > 0:
        plt.ylim(ylim)
    if color_by_cluster:
        cluster_parity = plot_clustering(burg, algorithm=cluster_algorithm, rhoH=rhoH, **cluster_args)
        plt.figure()
        plt.hist(cluster_parity, np.array(range(max(cluster_parity)+1)) + 0.5)
        plt.xlabel('$|\\vec{b}|^2$')
        plt.ylabel('Histogram for Burgers cluster sum')
        print(cluster_parity)
    return burg


def plt_cv(rhoH, *args, **kwargs):
    def mean(list_of_lists):
        max_len = max([len(x) for x in list_of_lists])
        sum = np.zeros(max_len)
        counter = np.zeros(max_len)
        for x in list_of_lists:
            for i in range(len(x)):
                counter[i] += 1
                sum[i] += x[i]
        return sum / counter
        # return list_of_lists[np.argmax([len(x) for x in list_of_lists])]

    default_plt_kwargs['linewidth'] = 4
    op_dir = op_path(rhoH, specif_op='Ising_k=4_undirected', *args, **kwargs)
    cvfiles, reals = sort_prefix(op_dir, 'Cv_vs_J')
    Cvs = []
    Js = []
    for file, realization in zip(cvfiles, reals):
        cv_mat = np.loadtxt(join(op_dir, file))
        sim = join(sims_dir, sim_name(rhoH, *args, **kwargs))
        z_path = join(sim, 'OP/Ising_k=4_undirected/z_' + str(realization) + '.txt')
        if os.path.exists(z_path):
            z = float(np.loadtxt(z_path))
        else:
            def ram_respecting_z_calc(sim, realization):
                """
                Help python remove the extremely heavy object ising_op from ram, as it sees the function where ising_op
                ref exist has ended, so it will remove it
                z = #bonds/#spheres~2 (not 4 because 4 is double counting the bonds)
                """
                ising_op = Ising(sim_path=sim, k_nearest_neighbors=4, directed=False)
                ising_op.update_centers(np.loadtxt(join(sim, str(realization))), realization)
                return ising_op.bonds_num / ising_op.N

            z = ram_respecting_z_calc(sim, realization)
            np.savetxt(z_path, np.array([z]))
        I = np.argsort(cv_mat[:, 0])
        cv_mat = cv_mat[I, :]
        J = cv_mat[:-1, 0] + np.diff(cv_mat[:, 0]) / 2
        dfdJ = np.diff(cv_mat[:, 2]) / np.diff(cv_mat[:, 0])
        Cvs.append(2 * J ** 2 * z * dfdJ)
        Js.append(J)
        # Cvs.append(cv_mat[:, 1])
        # Js.append(cv_mat[:, 0])

    Cv = mean(Cvs)
    J = mean(Js)
    i = np.argmax(Cv)
    if rhoH == 0.85:
        k = i + 1
        plt.plot(-J[:k], Cv[:k], colors_rho[rhoH], label="Solid " + prepare_lbl('rhoH=' + str(rhoH)),
                 **default_plt_kwargs)
        plt.plot(-J[k:], Cv[k:], colors_rho[rhoH], **default_plt_kwargs)
    elif rhoH == 0.8:
        plt.plot(-J[:i], Cv[:i], colors_rho[rhoH], label="Tetratic " + prepare_lbl('rhoH=' + str(rhoH)),
                 **default_plt_kwargs)
        plt.plot(-J[i:], Cv[i:], colors_rho[rhoH], **default_plt_kwargs)
    else:
        plt.plot(-J, Cv, colors_rho[rhoH], label="Liquid " + prepare_lbl('rhoH=' + str(rhoH)), **default_plt_kwargs)


def plt_anneal(rhoH, *args, **kwargs):
    default_plt_kwargs['linewidth'] = 2
    op_dir = op_path(rhoH, specif_op='Ising_k=4_undirected', *args, **kwargs)
    annealfiles, reals = sort_prefix(op_dir, 'anneal')
    anneal_mat, real = np.loadtxt(join(op_dir, annealfiles[1])), reals[1]
    # saving command in post process: np.savetxt(self.anneal_path, np.transpose([J] + self.frustration + self.Ms))
    J = anneal_mat[:, 0]
    anneal_reals = int((len(anneal_mat[0]) - 1) / 2)
    plt.plot(-J, anneal_mat[:, 1:anneal_reals], label='Anneal realization', **default_plt_kwargs)
    minf = np.min(anneal_mat[:, 1:anneal_reals])
    maxf = np.max(anneal_mat[-1, 1:anneal_reals])
    # plt.ylim([minf, minf * 1.05])
    return minf, maxf


def frustration(rhoH, frustration_realizations=5, *args, **kwargs):
    op_dir = op_path(rhoH, 'Graph', *args, **kwargs)
    frustrations = []
    for i, sp in enumerate(sort_prefix(op_dir, 'frustration_k=4_undirected_')[0]):
        if i >= frustration_realizations:
            break
        frustrations.append(np.loadtxt(join(op_dir, sp)))
    spheres_frustration = np.mean(frustrations)

    op_dir = op_path(rhoH, 'Ising_k=4_undirected', *args, **kwargs)
    cv_files, _ = sort_prefix(op_dir, 'Cv_vs_J_')
    frusts = []
    for cv_file in cv_files:
        _, Cv, f = np.loadtxt(join(op_dir, cv_file), unpack=True, usecols=(0, 1, 2))
        frusts.append(f[np.argmax(Cv)])
    f_argmaxcv = np.mean(frusts)

    anneal_file = sort_prefix(op_dir, 'anneal_')[0][0]
    A = np.loadtxt(os.path.join(op_dir, anneal_file))
    min_f = float('inf')
    reals = int((A.shape[1] - 1) / 2)
    for i in range(1, reals + 1):
        m = min(A[:, i])
        if m < min_f:
            min_f = m
    return min_f, spheres_frustration, f_argmaxcv


def plot_ising(rhos, rhoH_anneal, plot_heat_capcity=True, *args, **kwargs):
    plt.rcParams.update({'figure.figsize': (10, 10), 'axes.labelsize': 0.7 * size, 'xtick.labelsize': size * 0.5,
                         'ytick.labelsize': size * 0.5, 'legend.fontsize': size * 0.6})
    if not plot_heat_capcity:
        plt.rcParams.update({'figure.figsize': (8, 8)})
    plt.figure()

    if plot_heat_capcity:
        plt.subplot(211)
        for rhoH in rhos:
            plt_cv(rhoH, *args, **kwargs)
        plt.xlabel('$\\beta$J')
        plt.ylabel('$C_V$/$k_B$')
        plt.grid()
        plt.legend()
        plt.xlim([0.15, 1.45])

        plt.subplot(212)
    default_plt_kwargs['linewidth'] = 5
    N9e4dirs = [s for s in os.listdir(sims_dir) if s.startswith('N=9') and s.find('square') > 0]
    rhos = [float(re.split('(=|_)', dirname)[10]) for dirname in N9e4dirs]
    ground_state, sphere_frustration, max_cv_frustration = [], [], []
    for rho in rhos:
        try:
            a, b, c = frustration(rho)
        except Exception as err:
            print(err)
            a, b, c = np.nan, np.nan, np.nan
        ground_state.append(a)
        sphere_frustration.append(b)
        max_cv_frustration.append(c)
    rhos = np.array(rhos)

    def filter(I, rhos, ground_state, sphere_frustration, max_cv_frustration):
        return rhos[I], np.array(ground_state)[I], np.array(sphere_frustration)[I], np.array(max_cv_frustration)[I]

    rhos, ground_state, sphere_frustration, max_cv_frustration = filter(np.argsort(rhos), rhos, ground_state,
                                                                        sphere_frustration, max_cv_frustration)
    rhos, ground_state, sphere_frustration, max_cv_frustration = filter(np.logical_not(np.isnan(sphere_frustration)),
                                                                        rhos, ground_state, sphere_frustration,
                                                                        max_cv_frustration)
    plt.plot(rhos, ground_state, 'r', label='ground state', **default_plt_kwargs)
    plt.plot(rhos, sphere_frustration, 'm', label='sphere heights', **default_plt_kwargs)
    # plt.plot(rhos, max_cv_frustration, '.k', **default_plt_kwargs)
    # rhos_specific = [0.75, 0.8, 0.85]
    # max_cv_frustration_specific = [f for f, rho in zip(max_cv_frustration, rhos) if rho in rhos_specific]
    # plt.plot(rhos_specific, max_cv_frustration_specific, '.k', **default_plt_kwargs)
    rhos, ground_state, sphere_frustration, max_cv_frustration = filter(np.logical_not(np.isnan(max_cv_frustration)),
                                                                        rhos, ground_state, sphere_frustration,
                                                                        max_cv_frustration)
    if plot_heat_capcity:
        p = np.polyfit([-r for r in rhos] + [r for r in rhos], 2 * [f for f in max_cv_frustration], 3)
        # Weired fit with mirror around zero promise monotonic property to fit
        rhos_p = np.linspace(min(rhos), max(rhos))
        I = np.where(rhos_p < 0.78)
        plt.plot(rhos_p[I], np.polyval(p, rhos_p)[I], '--k', **default_plt_kwargs)
        I = np.where(rhos_p > 0.78)
        plt.plot(rhos_p[I], np.polyval(p, rhos_p[I]), '-k', label='argmax($C_V$) fit', **default_plt_kwargs)
        plt.plot(0.78, np.polyval(p, 0.78), '.k', markersize=20)

    plt.legend(loc=1)
    plt.grid()
    plt.ylabel('frustration')
    plt.xlabel(prepare_lbl('rhoH'))
    if plot_heat_capcity:
        plt.xlim([0.5, 0.9])
        plt.ylim([0, 0.35])
    else:
        plt.xlim([0.7, 0.9])
        plt.ylim([0, 0.16])
    plt.savefig('graphs/Ising' + ('' if plot_heat_capcity else '_without_heat_capacity'))

    plt.rcParams.update({'figure.figsize': (10, 7)})
    plt.figure()
    plt.grid()
    plt.xlabel('$\\beta$J')
    plt.ylabel('frustration')
    i = np.where(rhos == rhoH_anneal)[0][0]
    J_lim = [1, 3.0]
    plt.plot(J_lim, 2 * [sphere_frustration[i]], '--k', label='Sphere height', **default_plt_kwargs)
    minf_anneal, maxf_anneal = plt_anneal(rhoH=rhoH_anneal)
    minf = min(minf_anneal, sphere_frustration[i])
    maxf = max(maxf_anneal, sphere_frustration[i])
    plt.xlim(J_lim)
    plt.ylim([minf - (maxf - minf) / 5, maxf + (maxf - minf) / 5])
    plt.legend(loc=3)
    plt.savefig('graphs/Ising_anneal')
    return


def hist_local_psi(rhoH, rad=30, *args, **kwargs):
    default_plt_kwargs['linewidth'] = 5
    op_dir = op_path(rhoH, 'Local_psi_14', *args, **kwargs)
    files, reals = sort_prefix(op_dir, 'hist_rad=' + str(rad))
    rhos, psis = [], []
    for file in files:
        rho, psi = np.loadtxt(join(op_dir, file), unpack=True, usecols=(0, 1))
        rhos.append(rho)
        psis.append(psi)
    rho_avg = rhos[0]
    psi_avg = np.zeros(len(psis[0]))
    for rho, psi in zip(rhos, psis):
        psi_avg += np.interp(rho_avg, rho, psi)
    psi_avg /= len(psis)
    rho_avg = [0] + [r for r in rho_avg]
    psi_avg = [0] + [p for p in psi_avg]
    plt.plot(rho_avg, psi_avg, colors_rho[rhoH], label=prepare_lbl('rhoH=' + str(rhoH)), **default_plt_kwargs)


def plot_local_psi_hist(rhos, rad=30, *args, **kwargs):
    plt.rcParams.update(params)
    plt.rcParams.update({'figure.figsize': (10, 10)})
    plt.figure()
    for i, rho in enumerate(rhos):
        hist_local_psi(rho, rad, *args, **kwargs)
    plt.legend()
    plt.grid()
    plt.xlabel('|$\\psi_{4}^{r=' + str(int(rad / 2)) + '\sigma}$|')
    plt.ylabel('pdf')
    plt.savefig('graphs/local_psi_histogram')


def plot_magnetic_corr(rhos):
    default_plt_kwargs['linewidth'] = 5
    plt.rcParams.update({'figure.figsize': (12, 10)})
    plt.figure()
    corr_ylim = [1e-2, 1]
    corr_xlim = [0.8, 1e2]

    plot_corr(rhos, 'Bragg_Sm', poly_slope=1.0 / 3, single_slope_label=True)
    plt.legend([prepare_lbl('rhoH=' + str(r)) for r in rhos] + ['$x^{-1/3}$'])
    plt.ylim(corr_ylim)
    plt.xlim(corr_xlim)
    plt.xlabel('$\Delta$r/$\sigma$')
    plt.ylabel(prepare_lbl('Bragg_Sm'))
    plt.savefig('graphs/magnetic_bragg_corr')


def plot_bragg_peak(rhoH, *args, **kwargs):
    plt.rcParams.update({'figure.figsize': (10, 10), 'axes.labelsize': 0.45 * size, 'xtick.labelsize': size * 0.38,
                         'ytick.labelsize': size * 0.4, 'legend.fontsize': size * 0.4, 'axes.titlesize': size * 0.6})
    fig = plt.figure()
    axs = [fig.add_subplot(2, 1, 1, projection='3d'), fig.add_subplot(2, 1, 2, projection='3d')]
    for bragg_type, sub in zip(['Bragg_S', 'Bragg_Sm'], [0, 1]):
        try:
            op_dir = op_path(rhoH, bragg_type, *args, **kwargs)
            data_file = sort_prefix(op_dir, 'vec_')[0][0]
            kx, ky, S_values = np.loadtxt(join(op_dir, data_file), unpack=True, usecols=(0, 1, 2))
            ax = axs[sub]
            ax.scatter(kx, ky, S_values, '.')
            if sub == 0:
                ax.set_title(prepare_lbl('rhoH=' + str(rhoH)))
                ax.set_zlabel('$S(k)$')
            else:
                ax.set_zlabel('$S^M(k)$')
            ax.set_xlabel('$k_x$')
            ax.set_ylabel('$k_y$')
        except Exception as err:
            print(err)
    plt.savefig('graphs/Bragg_peak')


def plot_annealing(rhoH, real=None, *args, **kwargs):
    plt.figure()
    plt.rcParams.update(params)
    plt.rcParams.update({'figure.figsize': (12, 10)})
    default_plt_kwargs['linewidth'] = 2
    op_dir = op_path(rhoH, 'Ising_k=4_undirected', *args, **kwargs)
    if real is None:
        anneal_file = sort_prefix(op_dir, 'anneal_')[0][0]
    else:
        anneal_file = 'anneal_' + str(real) + '.txt'
    mat = np.loadtxt(join(op_dir, anneal_file))
    reals_count = int((mat.shape[1] - 1) / 2)  # first col is J, then Es and then Ms
    Es = [mat[:, j + 1] for j in range(reals_count)]
    J = mat[:, 0]
    for j in range(reals_count):
        if j == 0:
            plt.plot(J, Es[j], label="random initial condition realizations", **default_plt_kwargs)
        else:
            plt.plot(J, Es[j], **default_plt_kwargs)
    plt.xlabel('$\\beta$J')
    plt.ylabel('frustration')
    plt.legend()
    plt.grid()
    plt.ylim([0, 0.1])
    plt.savefig('graphs/anneal')


def converge_psi(rhoH, *args, **kwargs):
    op_dir = op_path(rhoH, specif_op='psi_14', *args, **kwargs)
    files, reals = sort_prefix(op_dir, 'vec_', reverse=False)
    psi_avg = []
    for file in files:
        psi_avg.append(np.abs(np.mean(np.loadtxt(join(op_dir, file), dtype=complex))))
    return psi_avg, reals


def plot_psi_convergence(rhos):
    plt.rcParams.update(params)
    plt.rcParams.update({'figure.figsize': (10, 10)})
    plt.figure()
    default_plt_kwargs['linewidth'] = 5

    def plt_rho(rho, label, s='-', **keywargs):
        psi_avg, reals = converge_psi(rho, **keywargs)
        plt.semilogx(reals, psi_avg, s + colors_rho[rho], label=label,
                     **default_plt_kwargs)

    for rho in rhos:
        plt_rho(rho, prepare_lbl('rhoH=' + str(rho)) + ', N=9e4, square ic')
        plt_rho(rho, 'N=4e4, honeycomb ic', s='--', initial_conditions='AF_triangle', N=4e4)
    plt.legend(loc=1)
    plt.xlabel('realization')
    plt.ylabel('$|\\overline{\\psi_{4}}|$')
    plt.ylim([0, 1.4])
    plt.grid()
    plt.savefig('graphs/psi_convergence')


def coarse_grain_burgers(rhoH, xlim=[], ylim=[], realization=None, max_pair_cleans_iterations=4,
                         clean_by_cluster_flag=True, cluster_algorithm=AgglomerativeClustering, distance_threshold=3,
                         cluster_args={'compute_full_tree': True, 'n_clusters': None, 'linkage': 'single'},
                         color_by_cluster=True, overwrite_cluster_clean=False):
    write_or_load = WriteOrLoad(join(sims_dir, sim_name(rhoH)))
    l_x, l_y, _, _, _, _, _, _ = write_or_load.load_Input()
    cyc = lambda p1, p2: cyc_dist(p1, p2, [l_x / 2, l_y / 2])
    cluster_args['affinity'] = lambda X: pairwise_distances(X, metric=cyc)
    cluster_args['distance_threshold'] = distance_threshold
    # for pair_cleans_iterations in range(max_pair_cleans_iterations + 1):
    #     quiver_cleaned_burgers(rhoH, xlim=xlim, ylim=ylim, realization=realization,
    #                            pair_cleans_iterations=pair_cleans_iterations, clean_by_cluster_flag=False)
    if clean_by_cluster_flag:
        # burg = quiver_cleaned_burgers(rhoH, xlim=xlim, ylim=ylim, realization=realization,
        #                               pair_cleans_iterations=max_pair_cleans_iterations, clean_by_cluster_flag=False)
        # plot_clustering(burg, algorithm=cluster_algorithm, rhoH=rhoH, **cluster_args)
        # plt.title('Clustering')
        quiver_cleaned_burgers(rhoH, xlim=xlim, ylim=ylim, realization=realization,
                               pair_cleans_iterations=max_pair_cleans_iterations,
                               clean_by_cluster_flag=True, cluster_algorithm=cluster_algorithm,
                               cluster_args=cluster_args,
                               color_by_cluster=color_by_cluster, overwrite_cluster_clean=overwrite_cluster_clean)


if __name__ == "__main__":
    # plot_pos_and_orientation([0.85, 0.83, 0.8], [0.8, 0.78, 0.77])
    rhoH, realization = 0.81, 94363239  # 0.8, 64155333  # 92549977  #
    # quiver_burger(rhoH, [85, 106.7], [126, 140.1], realization=realization, bonds=True, quiv=False)
    # quiver_burger(rhoH, [120, 170], [60, 90], realization=realization, bonds=True, quiv=True)
    xlim, ylim = [120, 170], [60, 90]
    # for rad in [None, 10, 5, 2, 0]:
    #     quiver_burger(rhoH, xlim, ylim, bonds=False, quiv=True, realization=realization, orientational_rad=rad)
    coarse_grain_burgers(rhoH, realization=realization, distance_threshold=5, overwrite_cluster_clean=True,
                         xlim=[0, 260], ylim=[0, 260], max_pair_cleans_iterations=0)

    # plot_ising([0.85, 0.8, 0.75], rhoH_anneal=0.8, plot_heat_capcity=True)

    # plot_local_psi_hist([0.8, 0.785, 0.78, 0.775, 0.77])
    # plot_magnetic_corr([0.85, 0.83, 0.8, 0.77])
    # plot_bragg_peak(0.8)
    # plot_annealing(0.8, real=45738368)
    # plot_psi_convergence([0.83, 0.8, 0.78])
    print('Finished succesfully')
#     TODO: recreate all graphs for rhoH=0.81, which looks much finer

plt.show()
