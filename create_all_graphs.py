import matplotlib.pyplot as plt
import numpy as np
import os, re
from sys import path as syspath

syspath.append('..\\OOP_hard_sphere_event_chain')
from post_process import Ising, Graph, PsiMN
from SnapShot import WriteOrLoad
from mpl_toolkits import mplot3d
import scipy
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import *

sims_dir = '..\\from_ATLAS3.0'
default_plt_kwargs = {'linewidth': 5, 'markersize': 10}
size = 30
params = {'legend.fontsize': size * 0.75, 'figure.figsize': (10, 10), 'axes.labelsize': size, 'axes.titlesize': size,
          'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75}
plt.rcParams.update(params)
colors_rho = {0.84: 'C0', 0.83: 'C1', 0.81: 'C2', 0.8: 'C9', 0.78: 'C3', 0.77: 'C4', 0.785: 'C5', 0.775: 'C6',
              0.75: 'C7', 0.79: 'C8', 0.71: 'C9', 0.85: 'gray', 0.82: 'yellow'}
direction_colors = {(1, 1): 'b', (1, -1): 'm', (-1, 1): 'g', (-1, -1): 'r', (0, 0): 'gray'}


def join(*args, **kwargs):
    return os.path.join(*args, **kwargs)


def sim_name(rhoH, N=9e4, h=0.8, initial_conditions='AF_square', algorithm='ECMC'):
    return 'N=' + str(int(N)) + '_h=' + str(h) + '_rhoH=' + str(rhoH) + '_' + initial_conditions + '_' + algorithm


def op_path(rhoH, specif_op=None, *args, **kwargs):
    op_dir = join(sims_dir, sim_name(rhoH, *args, **kwargs), 'OP')
    if specif_op is None:
        return op_dir
    else:
        return join(op_dir, specif_op)


def sort_prefix(folder_path, prefix='correlation_', surfix='.txt', reverse=True):
    relevent_files = [file for file in os.listdir(folder_path) if file.startswith(prefix) and file.endswith(surfix)]
    reals = [int(re.split('\.', re.split('_', file)[-1])[0]) for file in relevent_files]
    sorted_files = [f for _, f in sorted(zip(reals, relevent_files), reverse=reverse)]
    sorted_reals = sorted(reals, reverse=reverse)
    return sorted_files, sorted_reals


def prepare_lbl(lbl):
    lbl = re.sub('_', ' ', lbl)
    lbl = re.sub('rhoH', '$\\\\rho_H$', lbl)
    if lbl.startswith('psi'):
        for mn in ['14', '23', '16', '4', '6']:
            # lbl = re.sub('psi ' + mn, 'Orientational', lbl)
            lbl = re.sub('psi ' + mn, '$g_{' + mn + '}$', lbl)
    if lbl.startswith('Bragg Sm'):
        lbl = re.sub('Bragg Sm', '$g_k^M$', lbl)
    if lbl.startswith('Bragg S'):
        # lbl = re.sub('Bragg S', 'Positional', lbl)
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
        lbl = '$r^{-1/4\ or -1/3}$' if not single_slope_label else '$r{-' + str(np.round(poly_slope, 2)) + '}$'
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
    plt.legend([prepare_lbl('rhoH=' + str(r)) for r in rhos_pos] + ['$r^{-1/3}$'], loc=3)
    plt.loglog([60, 60], [0, 1], '--r', **default_plt_kwargs)
    # ax = plt.gca()
    # ax.legend_ = None

    plt.subplot(212)
    plot_corr(rhos_psi, 'psi_14', poly_slope=0.25)
    plt.ylim(corr_ylim)
    plt.xlim(corr_xlim)
    plt.ylabel(prepare_lbl('psi_4'))
    # plt.xlabel('$\Delta$r/$\sigma$')
    plt.xlabel('r/$\sigma$')
    plt.legend([prepare_lbl('rhoH=' + str(r)) for r in rhos_psi] + ['$r^{-1/4}$'], loc=3)
    # plt.legend([prepare_lbl('rhoH=' + str(r)) if r not in rhos_pos else '_nolegend_' for r in rhos_psi] + ['$r^{-1/4}$'], loc=3)
    plt.loglog([60, 60], [0, 1], '--r', **default_plt_kwargs)

    plt.savefig('graphs/orientation_and_position_corr')


def quiver_burger(rhoH, xlim, ylim, realization=None, bonds=True, frustrated_bonds=True, quiv=True,
                  orientational_rad=None, plot_centers=True, quiv_surfix='', new_fig=True, given_color=None, *args,
                  **kwargs):
    plt.rcParams.update({'figure.figsize': (15, 8)})
    if new_fig:
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
        file = 'vec_' + str(realization) + quiv_surfix + '.txt'
    spheres = np.loadtxt(join(sim, str(realization)))
    any_bonds = bonds or frustrated_bonds
    if any_bonds:
        op = Ising(sim_path=sim, k_nearest_neighbors=4, directed=False)
        op.update_centers(spheres, realization)
        spheres = op.spheres  # might reogranize order
    x = spheres[:, 0] / 2
    y = spheres[:, 1] / 2
    z = (spheres[:, 2] - np.mean(spheres[:, 2])) / 2
    if len(xlim) > 0 and len(ylim) > 0:
        up = np.array(
            [(z_ > 0) and xlim[0] < x_ < xlim[1] and ylim[0] < y_ < ylim[1] for (x_, y_, z_) in zip(x, y, z)])
        down = np.array(
            [(z_ <= 0) and xlim[0] < x_ < xlim[1] and ylim[0] < y_ < ylim[1] for (x_, y_, z_) in zip(x, y, z)])
    else:
        up = np.array([(z_ > np.mean(z)) for z_ in z])
        down = np.array([(z_ <= np.mean(z)) for z_ in z])
    if any_bonds:
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
                if spins[i] * spins[j] < 0 and bonds:
                    plt.plot(ex, ey, '--', color='lightgray')
    if plot_centers:
        # plt.plot(x[up], y[up], '.', color='orange', label='up', markersize=10)
        plt.plot(x[up], y[up], '.r', label='up', markersize=10)
        plt.plot(x[down], y[down], '.b', label='down', markersize=10)
    if quiv:
        burg = np.loadtxt(join(op_dir, file))
        if len(xlim) > 0 and len(ylim) > 0:
            I_box = np.array(
                [xlim[0] < x_ < xlim[1] and ylim[0] < y_ < ylim[1] for (x_, y_) in zip(burg[:, 0] / 2, burg[:, 1] / 2)])
            burg = burg[I_box, :]
        color = associate_color(np.sum(burg[:, 2:], 0)) if given_color is None else given_color
        plt.quiver(burg[:, 0] / 2, burg[:, 1] / 2, burg[:, 2] / 2, burg[:, 3] / 2, angles='xy', scale_units='xy',
                   scale=1, label='Burger field', width=3e-3, zorder=7, color=color)

    plt.axis('equal')
    # plt.legend()

    # x1, x2 = xlim
    # y1, y2 = ylim
    # plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], '-k', **default_plt_kwargs)

    plt.savefig('graphs/burger_vectors')
    return


def cyc_dist(p1, p2, boundaries):
    dx = np.array(p1) - p2  # direct vector
    dsq = 0
    for i in range(2):
        L = boundaries[i]
        dsq += min(dx[i] ** 2, (dx[i] + L) ** 2, (dx[i] - L) ** 2)  # find shorter path through B.D.
    return np.sqrt(dsq)


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
    elif 0.78 <= rhoH <= 0.83:
        plt.plot(-J[:i], Cv[:i], colors_rho[rhoH], label="Tetratic " + prepare_lbl('rhoH=' + str(rhoH)),
                 **default_plt_kwargs)
        plt.plot(-J[i:], Cv[i:], colors_rho[rhoH], **default_plt_kwargs)
    else:
        plt.plot(-J, Cv, colors_rho[rhoH], label="Liquid " + prepare_lbl('rhoH=' + str(rhoH)), **default_plt_kwargs)


def plt_anneal(rhoH, *args, **kwargs):
    default_plt_kwargs['linewidth'] = 2
    op_dir = op_path(rhoH, specif_op='Ising_k=4_undirected', *args, **kwargs)
    annealfiles, reals = sort_prefix(op_dir, 'anneal')
    anneal_mat, real = np.loadtxt(join(op_dir, annealfiles[0])), reals[0]
    # saving command in post process: np.savetxt(self.anneal_path, np.transpose([J] + self.frustration + self.Ms))
    J = anneal_mat[:, 0]
    # anneal_reals = int((len(anneal_mat[0]) - 1) / 2)
    anneal_reals = int((anneal_mat.shape[1] - 1) / 2)  # first col is J, then Es and then Ms
    plt.plot(-J, anneal_mat[:, 1:anneal_reals + 1], label='Anneal realization', **default_plt_kwargs)
    minf = np.min(anneal_mat[:, 1:anneal_reals + 1])
    maxf = np.max(anneal_mat[-1, 1:anneal_reals + 1])
    # plt.ylim([minf, minf * 1.05])
    return minf, maxf


def frustration(rhoH, *args, **kwargs):
    op_dir = op_path(rhoH, 'Graph', *args, **kwargs)
    spheres_frustration = np.loadtxt(join(op_dir, sort_prefix(op_dir, 'frustration_k=4_undirected_')[0][0]))

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


def plot_ising(rhos_heat_capacity=None, plot_heat_capcity=True, *args, **kwargs):
    plt.rcParams.update({'figure.figsize': (10, 10), 'axes.labelsize': 0.7 * size, 'xtick.labelsize': size * 0.6,
                         'ytick.labelsize': size * 0.6, 'legend.fontsize': 0.7 * size})
    if not plot_heat_capcity:
        plt.rcParams.update({'figure.figsize': (12, 6)})
    plt.figure()

    if plot_heat_capcity:
        plt.subplot(211)
        for rhoH in rhos_heat_capacity:
            plt_cv(rhoH, *args, **kwargs)
        plt.xlabel('$\\beta$J')
        plt.ylabel('$C_V$/$k_B$')
        plt.grid()
        plt.legend()
        plt.xlim([0.15, 1.45])

        plt.subplot(212)
    default_plt_kwargs['linewidth'] = 5
    # N9e4dirs = [s for s in os.listdir(sims_dir) if s.startswith('N=90000') and s.find('square') > 0]
    # rhos = [float(re.split('(=|_)', dirname)[10]) for dirname in N9e4dirs]
    rhos = [0.6] + [np.round(r, 2) for r in np.linspace(0.71, 0.9, 20)]
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
    plt.plot(rhos, ground_state, 'k', label='$f_\mathrm{frust}$', **default_plt_kwargs)
    plt.plot(rhos, sphere_frustration, 'b', label='$f_\mathrm{unsat}$', **default_plt_kwargs)
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
    # plt.ylabel('$f_\mathrm{unsat}$')
    plt.xlabel(prepare_lbl('rhoH'))
    if plot_heat_capcity:
        plt.xlim([0.5, 0.9])
        plt.ylim([0, 0.35])
    else:
        plt.xlim([0.7, 0.9])
        plt.ylim([0, 0.16])
    for rho in [0.77, 0.785, 0.83]:
        plt.plot(2 * [rho], [0, 1], '--g', linewidth=5, zorder=0)
    ytext = 0.125
    text_args = {'c': 'g', 'size': 25, 'weight': 'bold'}
    plt.text(0.835, ytext, 'Solid', **text_args)
    plt.text(0.79, ytext, 'Tetratic', **text_args)
    plt.text(0.74, ytext, 'Liquid', **text_args)
    plt.savefig('graphs/Ising' + ('' if plot_heat_capcity else '_without_heat_capacity') + '_with_phases')

    # plt.rcParams.update({'figure.figsize': (10, 7)})
    # plt.figure()
    # plt.grid()
    # plt.xlabel('$\\beta$J')
    # plt.ylabel('$f_\mathrm{unsat}$')
    # i = np.where(rhos == rhoH_anneal)[0][0]
    # J_lim = [1, 3.0]
    # plt.plot(J_lim, 2 * [sphere_frustration[i]], '--k', label='Sphere height', **default_plt_kwargs)
    # minf_anneal, maxf_anneal = plt_anneal(rhoH=rhoH_anneal)
    # minf = min(minf_anneal, sphere_frustration[i])
    # maxf = max(maxf_anneal, sphere_frustration[i])
    # plt.xlim(J_lim)
    # plt.ylim([minf - (maxf - minf) / 5, maxf + (maxf - minf) / 5])
    # plt.legend(loc=3)
    # plt.savefig('graphs/Ising_anneal')
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
    plt.ylabel('PDF')
    plt.savefig('graphs/local_psi_histogram')


def plot_magnetic_corr(rhos):
    default_plt_kwargs['linewidth'] = 5
    plt.rcParams.update({'figure.figsize': (12, 9)})
    plt.figure()
    corr_ylim = [1e-2, 1]
    corr_xlim = [0.8, 1e2]

    plot_corr(rhos, 'Bragg_Sm', poly_slope=1.0 / 6, single_slope_label=True)
    plt.legend([prepare_lbl('rhoH=' + str(r)) for r in rhos] + ['$r^{-1/6}$'])
    plt.ylim(corr_ylim)
    plt.xlim(corr_xlim)
    plt.xlabel('r/$\sigma$')
    plt.ylabel(prepare_lbl('Bragg_Sm'))
    plt.loglog([60, 60], [0, 1], '--r', **default_plt_kwargs)
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


def plot_color_bargg_peaks(rhoH_vec, bragg_type='Bragg_S', save=False, load=True, *args, **kwargs):
    kres = 50
    anglesres = 500
    zero_k_area = 15000

    plt.rcParams.update({'figure.figsize': (10, 10), 'axes.labelsize': 0.45 * size, 'xtick.labelsize': size * 0.38,
                         'ytick.labelsize': size * 0.4, 'legend.fontsize': size * 0.4, 'axes.titlesize': size * 0.6})
    fig = plt.figure()
    for i, rhoH in enumerate(rhoH_vec):
        plt.subplot(np.ceil(len(rhoH_vec) / 2), 2, i + 1)
        if (i + 1) % 2 == 1:
            plt.ylabel('$\\frac{a_0}{2\pi}k_y$')
        if i + 1 > 2:
            plt.xlabel('$\\frac{a_0}{2\pi}k_x$')
        plt.title('$\\rho_H$=' + str(rhoH))

        sim_path = join(sims_dir, sim_name(rhoH, *args, **kwargs))
        write_or_load = WriteOrLoad(sim_path)
        l_x, l_y, _, _, _, _, _, _ = write_or_load.load_Input()
        N = 90000
        a_0 = np.sqrt(l_x * l_y / N)
        if bragg_type == 'Bragg_S':
            k_ideal = 2 * np.pi / a_0
        elif bragg_type == 'Bragg_Sm':
            k_ideal = np.sqrt(2) * np.pi / a_0

        load_save_path = join(op_path(rhoH, bragg_type, *args, **kwargs), 'local_data')
        if load and (not os.path.exists(load_save_path)):
            load = False
        if load:
            kx, ky, S = np.loadtxt(load_save_path, unpack=True, usecols=(0, 1, 2))
        else:
            psi = PsiMN(sim_path, 1, 4)
            psi.read_or_calc_write()
            print(psi.spheres_ind)
            _, sp = psi.rotate_spheres()
            sp = np.array(sp)
            x = sp[:, 0]
            y = sp[:, 1]
            z = sp[:, 2]
            z = (z - np.mean(z)) * 2 / (np.max(z) - np.min(z))
            kmin, kmax = 0.5 * k_ideal, 1.5 * k_ideal
            k_rads = np.linspace(kmin, kmax, kres)
            angles = np.linspace(0, 2 * np.pi, anglesres)
            k = [np.array([k_ * np.cos(t), k_ * np.sin(t)]) for k_ in k_rads for t in angles]
            k_inner = np.linspace(0, kmin, zero_k_area)
            ts = [np.random.random() * np.pi * 2 for _ in k_inner]
            i_inner = len(k)
            k = k + [np.array([k_ * np.cos(t), k_ * np.sin(t)]) for (k_, t) in zip(k_inner, ts)]
            S = np.zeros(len(k))
            kx = np.array([k[0] for k in k])
            ky = np.array([k[1] for k in k])
            for i, k_ in enumerate(k):
                phase = k_[0] * x + k_[1] * y
                if bragg_type == 'Bragg_S':
                    S[i] = 1 / N * (np.sum(np.cos(phase)) ** 2 + np.sum(np.sin(phase)) ** 2)
                elif bragg_type == 'Bragg_Sm':
                    S[i] = 1 / N * (np.sum(z * np.cos(phase)) ** 2 + np.sum(z * np.sin(phase)) ** 2)
            try:
                op_dir = op_path(rhoH, bragg_type, *args, **kwargs)
                data_file = 'vec_' + str(psi.spheres_ind) + '.txt'
                kxs_precaculated, kys_precaculated, S_precaculated = np.loadtxt(join(op_dir, data_file), unpack=True,
                                                                                usecols=(0, 1, 2))
                I = np.where([k_ < 1.5 * k_ideal for k_ in np.sqrt(kxs_precaculated ** 2 + kys_precaculated ** 2)])
                if max(S[:i_inner]) > max(S_precaculated):
                    print("Found Higher S in current analysis by a factor of " + str(max(S) / max(S_precaculated)))
                else:
                    print("Higher S in ATLAS analysis")
                S = np.concatenate((S, S_precaculated[I]))
                kx = np.concatenate((kx, kxs_precaculated[I]))
                ky = np.concatenate((ky, kys_precaculated[I]))
            except:
                pass
            idx = S.argsort()
            kx, ky, S = kx[idx], ky[idx], S[idx]
        vmax = np.max([S_ for (k_, S_) in zip(np.sqrt(kx ** 2 + ky ** 2), S) if k_ > 0.5 * k_ideal])
        # plt.scatter(a_0 * kx / (2 * np.pi), a_0 * ky / (2 * np.pi), c=S, s=3, vmin=0, vmax=vmax)
        plt.scatter(a_0 * kx / (2 * np.pi), a_0 * ky / (2 * np.pi), c=np.log(S), s=3, vmin=0, vmax=np.log(vmax))
        if save:
            # append([k[0], k[1], S_])
            np.savetxt(load_save_path,
                       np.array([[kx[i], ky[i], S[i]] for i in range(len(S))]))
    plt.savefig('graphs/color_' + bragg_type)


def gradients_around_optimal_k(rhoH_vec, bragg_type='Bragg_Sm', overwrite=False):
    plt.rcParams.update({'figure.figsize': (20, 10), 'axes.labelsize': 0.6 * size, 'xtick.labelsize': size * 0.6,
                         'ytick.labelsize': size * 0.6, 'legend.fontsize': size * 0.6, 'axes.titlesize': size * 0.6})
    fig = plt.figure()
    res = 501
    k_factor_span = 0.02
    angle_span = 0.1
    for rhoH in rhoH_vec:
        sim_path = join(sims_dir, sim_name(rhoH))
        psi = PsiMN(sim_path, 1, 4)
        psi.read_or_calc_write()
        print(psi.spheres_ind)
        _, sp = psi.rotate_spheres()
        sp = np.array(sp)
        x = sp[:, 0]
        y = sp[:, 1]
        z = sp[:, 2]
        z = (z - np.mean(z)) * 2 / (np.max(z) - np.min(z))
        N = 90000

        def S_func(k):
            phase = k[0] * x + k[1] * y
            if bragg_type == 'Bragg_S':
                S = 1 / N * (np.sum(np.cos(phase)) ** 2 + np.sum(np.sin(phase)) ** 2)
            elif bragg_type == 'Bragg_Sm':
                S = 1 / N * (np.sum(z * np.cos(phase)) ** 2 + np.sum(z * np.sin(phase)) ** 2)
            return S

        load_save_path = join(op_path(rhoH, bragg_type), 'local_data')
        kx, ky, S = np.loadtxt(load_save_path, unpack=True, usecols=(0, 1, 2))
        i_max = np.argmax(S)
        k_rad_max = np.sqrt(kx[i_max] ** 2 + ky[i_max] ** 2)
        k_angle = np.arctan2(ky[i_max], kx[i_max])

        load_radial_azimuthal_path = join(op_path(rhoH, bragg_type), 'radial_azimuthal_local_data')
        if (not overwrite) and os.path.exists(load_radial_azimuthal_path):
            angles, k_rads, S_angles, S_rads = np.loadtxt(load_radial_azimuthal_path, unpack=True, usecols=(0, 1, 2, 3))
        else:
            angles = k_angle + np.linspace(-angle_span, angle_span, res)
            S_angles = []
            for angle in angles:
                S_angles.append(S_func([k_rad_max * np.cos(angle), k_rad_max * np.sin(angle)]))
            k_rads = k_rad_max * np.linspace(1 - k_factor_span, 1 + k_factor_span, res)
            S_rads = []
            for k_rad in k_rads:
                S_rads.append(S_func([k_rad * np.cos(k_angle), k_rad * np.sin(k_angle)]))
            np.savetxt(load_radial_azimuthal_path,
                       np.array([[angles[i], k_rads[i], S_angles[i], S_rads[i]] for i in range(res)]))

        plt.subplot(2, 1, 1)
        plt.xlabel('Azimuthal deviation from maximum [radians]')
        plt.ylabel('$S$/$S_{max}$')
        plt.grid('on')
        plt.semilogy(angles - k_angle, np.array(S_angles) / np.max(S),
                     label='$\\rho_H=$' + str(rhoH) + ', $S_{max}=$' + str(int(np.max(S))))
        plt.ylim([1 / N, 1])
        plt.legend(loc=1)

        plt.subplot(2, 1, 2)
        plt.xlabel('Radial factor from maximum $|k|$/$|k_{max}|$')
        plt.ylabel('$S$/$S_{max}$')
        plt.grid('on')
        plt.semilogy(k_rads / k_rad_max, np.array(S_rads) / np.max(S),
                     label='$\\rho_H=$' + str(rhoH) + ', $S_{max}=$' + str(int(np.max(S))))
        plt.ylim([1 / N, 1])
    plt.savefig('graphs/gradients_' + bragg_type)


def plot_z_histogram(rhoH_vec, *args, **kwargs):
    plt.rcParams.update(params)
    plt.rcParams.update({'figure.figsize': (10, 10)})
    plt.figure()
    for rhoH in rhoH_vec:
        sim_path = join(sims_dir, sim_name(rhoH, *args, **kwargs))
        write_or_load = WriteOrLoad(sim_path)
        sp = write_or_load.last_spheres()[0]
        z = sp[:, 2]
        z = (z - np.mean(z)) * 2 / (np.max(z) - np.min(z))
        hist, bin_edges = np.histogram(z, 50, density=True)
        plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist, colors_rho[rhoH], label='$\\rho_H$=' + str(rhoH),
                 linewidth=3)
    plt.xlabel('normalized sphere height')
    plt.ylabel('PDF')
    plt.grid()
    plt.legend()
    plt.savefig('graphs/z_histogram')


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


def plot_psi_convergence(rhos, orientation=False):
    # params = {'legend.fontsize': size * 0.75, 'figure.figsize': (10, 10), 'axes.labelsize': size,
    #           'axes.titlesize': size,
    #           'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75}
    plt.rcParams.update(params)
    plt.rcParams.update({'figure.figsize': (10 if not orientation else 20, 10), 'legend.fontsize': size * 0.6})
    plt.figure()
    default_plt_kwargs['linewidth'] = 5

    def plt_rho(rho, label, s='-', **keywargs):
        fname = join(op_path(rho, specif_op='psi_14', **keywargs), 'mean_vs_real.txt')
        reals, psi_avg = np.loadtxt(fname, dtype=complex, unpack=True, usecols=(0, 1))
        if not orientation:
            reals, psi_avg = np.real(reals), np.abs(psi_avg)
            plot_func = plt.semilogx
        else:
            reals, psi_avg = np.real(reals), np.real(np.log(psi_avg / np.abs(psi_avg)) / 4j)
            span = np.max(np.abs(psi_avg))
            psi_avg = psi_avg / span
            label += ', span=' + str(np.round(span, 4))
            plot_func = plt.plot
        plot_func(reals, psi_avg, s, color=colors_rho[rho], label=label, **default_plt_kwargs)

    for rho in rhos:
        rho_str = str(rho) if rho != 0.8 else '0.80'
        plt_rho(rho, prepare_lbl('rhoH=' + rho_str) + (', square ic, N=90,000' if not orientation else ''))
        # plt_rho(rho, '        honeycomb ic, N=40,000', s='--', initial_conditions='AF_triangle', N=4e4)
        if not orientation:
            plt_rho(rho, 'honeycomb ic, N=40,000', s='--', initial_conditions='AF_triangle', N=4e4)
        # plt_rho(rho, prepare_lbl('rhoH=' + str(rho)) + ', square ic')
        # plt_rho(rho, 'honeycomb ic', s='--', initial_conditions='AF_triangle')
    plt.legend(loc=1 if not orientation else 0)
    plt.xlabel('realization')
    plt.ylabel(
        '$|\\overline{\\psi_{4}}|$' if not orientation else 'Normalized sample orientation $\\theta=\\frac{1}{4i}log' +
                                                            '\\left(\\frac{\\overline{\\psi_{4}}}{|\\overline' +
                                                            '{\\psi_{4}}|}\\right)$')
    if not orientation:
        plt.ylim([0, 1.4])
    plt.grid()
    plt.savefig('graphs/' + ('psi_convergence' if not orientation else 'sample_orientation'))


def undirected_pairs_graph(X, l_x, l_y):
    cyc = lambda p1, p2: cyc_dist(p1, p2, [l_x, l_y])
    graph = kneighbors_graph(X, n_neighbors=1, metric=cyc)
    I, J, _ = scipy.sparse.find(graph)[:]
    Ed = [(i, j) for (i, j) in zip(I, J)]
    Eud = []
    N = len(X)
    udgraph = scipy.sparse.csr_matrix((N, N))
    for i, j in Ed:
        if ((j, i) in Ed) and ((i, j) not in Eud) and ((j, i) not in Eud):
            Eud.append((i, j))
            udgraph[i, j] = 1
            udgraph[j, i] = 1
    graph = udgraph
    nearest_neighbor = [[j for j in graph.getrow(i).indices] for i in range(N)]
    return nearest_neighbor


def clean_bound_pairs(burg, sim_path, cutoff=np.inf):
    # save graph into real num with 000 to differentiate from usual graph saves.
    write_or_load = WriteOrLoad(sim_path)
    l_x, l_y, _, _, _, _, _, _ = write_or_load.load_Input()
    nearest_neighbor = undirected_pairs_graph(burg[:, :2], l_x, l_y)
    cleaned_burg = []
    for i in range(len(burg)):
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
        if cyc_dist(r1, r2, [l_x, l_y]) > cutoff or np.linalg.norm(b1 + b2) > 1e-5:
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
        if np.linalg.norm(np.sum(burg[cluster, 2:], 0)) > 1e-5:
            for dislocation_ind in cluster:
                clean_burgers.append([burg[dislocation_ind, i] for i in range(4)])
    return np.array(clean_burgers)


def plot_clustering(burg, algorithm, *args, **kwargs):
    X = burg[:, :2]
    model = algorithm(*args, **kwargs)
    try:
        model.fit(X)
        yhat = model.predict(X)
    except AttributeError:
        yhat = model.fit_predict(X)
    clusters = np.unique(yhat)
    cluster_parity = []
    a = min([np.linalg.norm(b) for b in burg[:, 2:]])
    for cluster in clusters:
        row_ix = np.where(yhat == cluster)[0]
        sum_b = np.sum(burg[row_ix, 2:4], 0)
        color = associate_color(np.sum(burg[row_ix, 2:4], 0))
        cluster_parity.append(int(np.round(np.linalg.norm(sum_b / a) ** 2)))
        if cluster_parity[-1] % 2 == 0:
            plt.plot(X[row_ix, 0] / 2, X[row_ix, 1] / 2, '.', markersize=15, color=color)
        else:
            plt.plot(X[row_ix, 0] / 2, X[row_ix, 1] / 2, '*', markersize=10)
    return np.array(cluster_parity)


def quiver_cleaned_burgers(rhoH=None, realization=None, pair_cleans_iterations=1, cutoff=np.inf, xlim=[], ylim=[],
                           overwrite_pair_cleaning=False, clean_by_cluster_flag=False, cluster_algorithm=None,
                           cluster_args={}, overwrite_cluster_clean=True, color_by_cluster=False,
                           pair_cluster_iterations=1, sim_folder=None, plot_for_paper=False, box=[], arrow_width=3e-3,
                           *args, **kwargs):
    plt.rcParams.update({'figure.figsize': (15, 8)})
    plt.figure()
    sim = join(sims_dir, sim_name(rhoH, *args, **kwargs) if sim_folder is None else sim_folder)
    op_dir = join(sim, 'OP', 'burger_vectors')
    if realization is None:
        files, reals = sort_prefix(op_dir, 'vec_')
        realization = reals[0]
        file = files[0]
    else:
        file = 'vec_' + str(realization) + '.txt'
    plt.axis('equal')
    # plt.legend()
    burg = np.loadtxt(join(op_dir, file))
    if plot_for_paper:
        # plt.quiver(burg[:, 0] / 2, burg[:, 1] / 2, burg[:, 2] / 2, burg[:, 3] / 2, angles='xy', scale_units='xy',
        #            scale=1, label='Burger field', width=3e-3, color='gray')
        plt.plot(burg[:, 0] / 2, burg[:, 1] / 2, '.', markersize=2, color='gray')
        # Fig 3 gray small dots of microscopic Burgers field
    name = 'vec_' + str(realization)
    for k in range(pair_cluster_iterations):
        nom_name = name
        for i in range(pair_cleans_iterations):
            name = nom_name + '_paired-' + str(i + 1)
            clean_path = join(op_dir, name + '.txt')
            if os.path.exists(clean_path) and not overwrite_pair_cleaning:
                burg = np.loadtxt(clean_path)
            else:
                burg = clean_bound_pairs(burg, sim, cutoff=cutoff)
                np.savetxt(clean_path, burg)
        if clean_by_cluster_flag:
            name += '_clustered-D=' + str(int(cluster_args['distance_threshold']))
            clean_path = join(op_dir, name + '.txt')
            if os.path.exists(clean_path) and (not overwrite_cluster_clean):
                burg = np.loadtxt(clean_path)
            else:
                burg = clean_by_cluster(burg, cluster_algorithm, **cluster_args)
                np.savetxt(clean_path, burg)
    if len(box) > 0:
        xlim_box, ylim_box = box
        x1, x2 = xlim_box
        y1, y2 = ylim_box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], '-k', **default_plt_kwargs)
    if not plot_for_paper:
        plt.quiver(burg[:, 0] / 2, burg[:, 1] / 2, burg[:, 2] / 2, burg[:, 3] / 2, angles='xy', scale_units='xy',
                   scale=1,
                   label='Burger field', width=arrow_width, zorder=7)
    plt.title(('Microscopic Burgers' if pair_cleans_iterations == 0
               else ('Pair Cleaned ' + str(pair_cleans_iterations) + ' times')) + \
              ('' if (not clean_by_cluster_flag) else ', cluster cleaned D=' + str(int(
                  cluster_args['distance_threshold'] / 2)) + '$\sigma$') + (
                  '' if pair_cluster_iterations == 1 else 'pairs+cluster iterated ' + str(pair_cluster_iterations)))
    if len(xlim) > 0:
        plt.xlim(xlim)
    if len(ylim) > 0:
        plt.ylim(ylim)
    if color_by_cluster:
        # Fig 3 colored dots
        cluster_parity = plot_clustering(burg, algorithm=cluster_algorithm, **cluster_args)
        if plot_for_paper:
            plt.savefig('graphs/global_Burgers_field_w_cleaning_and_zoom_in')
        plt.figure()
        plt.hist(cluster_parity, np.array(range(max(cluster_parity) + 1)) + 0.5)
        plt.xlim([0, 9])
        plt.xlabel('$|\\vec{b}|^2$')
        plt.ylabel('Histogram for Burgers cluster sum')
        print(cluster_parity)
    return burg


def coarse_grain_burgers(rhoH=None, xlim=[], ylim=[], realization=None, max_pair_cleans_iterations=4,
                         clean_by_cluster_flag=True, cluster_algorithm=AgglomerativeClustering, distance_threshold=10,
                         cluster_args={'compute_full_tree': True, 'n_clusters': None, 'linkage': 'single'},
                         color_by_cluster=True, overwrite_cluster_clean=False, sim_folder=None, plot_for_paper=False,
                         box=[]):
    write_or_load = WriteOrLoad(join(sims_dir, sim_name(rhoH) if sim_folder is None else sim_folder))
    l_x, l_y, _, _, _, _, _, _ = write_or_load.load_Input()
    cyc = lambda p1, p2: cyc_dist(p1, p2, [l_x, l_y])
    cluster_args['affinity'] = lambda X: pairwise_distances(X, metric=cyc)
    cluster_args['distance_threshold'] = distance_threshold
    name = None
    kwargs = {'rhoH': rhoH, 'xlim': xlim, 'ylim': ylim, 'realization': realization,
              'clean_by_cluster_flag': False, 'cluster_algorithm': cluster_algorithm, 'cluster_args': cluster_args,
              'overwrite_cluster_clean': overwrite_cluster_clean, 'sim_folder': sim_folder}
    if plot_for_paper:
        kwargs['pair_cleans_iterations'] = max_pair_cleans_iterations
    else:
        for pair_cleans_iterations in range(max_pair_cleans_iterations + 1):
            kwargs['pair_cleans_iterations'] = pair_cleans_iterations
            burg = quiver_cleaned_burgers(**kwargs)
    if clean_by_cluster_flag:
        if not plot_for_paper:
            quiver_cleaned_burgers(**kwargs)
            plot_clustering(burg, algorithm=cluster_algorithm, **cluster_args)
            plt.title('Clustering D=' + str(np.round(distance_threshold / 2, 1)) + '$\sigma$')
        kwargs['pair_cleans_iterations'] = max_pair_cleans_iterations
        kwargs['clean_by_cluster_flag'] = True
        kwargs['color_by_cluster'] = color_by_cluster
        kwargs['plot_for_paper'] = plot_for_paper
        kwargs['box'] = box
        quiver_cleaned_burgers(**kwargs)
        # kwargs['pair_cluster_iterations'] = 2
        # quiver_cleaned_burgers(**kwargs)


def coarse_grain_null_model(ref_rhoH, ref_real, pair_cleans=4, clean_by_cluster_flag=True, distance_threshold=10,
                            bound_pairs=True, *args, **kwargs):
    burg_file_surfix = 'ref-rhoH=' + str(ref_rhoH) + '_real=' + str(ref_real) + '_bound-pairs' if bound_pairs else ''
    burg_name = 'vec_' + burg_file_surfix + '.txt'
    sim_folder = 'null_models'
    burg_dir_path = join(sims_dir, sim_folder, 'OP', 'burger_vectors')
    load = WriteOrLoad(join(sims_dir, sim_name(ref_rhoH, *args, **kwargs)))
    l_x, l_y, l_z, _, _, _, _, _ = load.load_Input()
    load.boundaries = [l_x, l_y, l_z]
    if not os.path.exists(join(burg_dir_path, burg_name)):
        op_dir = op_path(ref_rhoH, 'burger_vectors')
        file = 'vec_' + str(ref_real) + '.txt'
        ref_burg = np.loadtxt(join(op_dir, file))
        a = min([np.linalg.norm(b) for b in ref_burg[:, 2:]])
        if not bound_pairs:
            N_dislocations = ref_burg.shape[0]
            burg = gen_rand_dislocations(N_dislocations, l_x, l_y, a)
        else:
            neighbors = undirected_pairs_graph(ref_burg[:, :2], l_x, l_y)
            dists = []
            for i, neighbor_list in enumerate(neighbors):
                if len(neighbor_list) == 0:
                    continue
                j = neighbor_list[0]
                if j < i or (np.linalg.norm(ref_burg[i, 2:] + ref_burg[j, 2:]) > 1e-5):
                    continue
                dists.append(cyc_dist(ref_burg[i, :2], ref_burg[j, :2], [l_x, l_y]))
            N_pairs = len(dists)
            pairs_mean_separation = np.mean(dists)
            pairs_std_separation = np.std(dists)
            hist, bin_edges = np.histogram(dists, density=True, bins=np.linspace(0.3, 5, 48))
            bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
            plt.rcParams.update(params)
            plt.figure()
            plt.plot(bin_centers / 2, hist, label='From $\\rho_H$=0.81 simulation', **default_plt_kwargs)
            x = np.linspace(pairs_mean_separation - 3 * pairs_std_separation,
                            pairs_mean_separation + 3 * pairs_std_separation, 100)
            plt.plot(x / 2, scipy.stats.norm.pdf(x, pairs_mean_separation, pairs_std_separation),
                     label='Calibrated normal pdf', **default_plt_kwargs)
            # plt.plot(x, scipy.stats.expon.pdf(x, scale=pairs_mean_separation), label='exponential pdf')
            plt.grid()
            plt.xlabel('Separation/$\sigma$')
            plt.ylabel('PDF')
            plt.legend()

            ref_burg_clustered_name = 'vec_' + str(realization) + '_paired-4_clustered-D=10.txt'
            ref_burg_clustered = np.loadtxt(join(op_dir, ref_burg_clustered_name))
            X = ref_burg_clustered[:, :2]
            cyc = lambda p1, p2: cyc_dist(p1, p2, [l_x, l_y])
            model = AgglomerativeClustering(compute_full_tree=True, n_clusters=None, linkage='single',
                                            distance_threshold=distance_threshold,
                                            affinity=lambda X: pairwise_distances(X, metric=cyc))
            try:
                model.fit(X)
                yhat = model.predict(X)
            except AttributeError:
                yhat = model.fit_predict(X)
            N_singles = len(np.unique(yhat))

            print((N_pairs, N_singles, pairs_mean_separation, pairs_std_separation, l_x, l_y, a))
            burg = gen_dislocations_w_pairs(N_pairs, N_singles, pairs_mean_separation, pairs_std_separation, l_x, l_y,
                                            a)
        if not os.path.exists(burg_dir_path):
            os.makedirs(burg_dir_path)
        np.savetxt(join(burg_dir_path, burg_name), np.array(burg))
    load.output_dir = join(sims_dir, sim_folder)
    load.save_Input(rad=0, rho_H=0, edge=0, n_row=0, n_col=0)
    coarse_grain_burgers(xlim=[0, l_x / 2], ylim=[0, l_y / 2], distance_threshold=distance_threshold,
                         overwrite_cluster_clean=False, max_pair_cleans_iterations=pair_cleans,
                         clean_by_cluster_flag=clean_by_cluster_flag, sim_folder=sim_folder,
                         realization=burg_file_surfix)


def gen_rand_dislocations(N_dislocations, l_x, l_y, a):
    burg = []
    directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    for i in range(N_dislocations):
        direction = directions[np.random.randint(len(directions))]
        pos = np.random.uniform(high=(l_x, l_y))
        burg.append([x for x in pos] + [d * a for d in direction])
    return burg


def gen_dislocations_w_pairs(N_pairs, N_singles, pairs_mean_separation, pairs_std_separation, l_x, l_y, a):
    burg = []
    directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    for i in range(N_pairs):
        direction = directions[np.random.randint(len(directions))]
        pos = np.random.uniform(high=(l_x, l_y))
        t = np.random.uniform(high=2 * np.pi)
        pos2 = pos + (pairs_mean_separation + np.random.normal() * pairs_std_separation) * np.array(
            [np.cos(t), np.sin(t)])
        burg.append([x for x in pos] + [d * a for d in direction])
        burg.append([x for x in pos2] + [-d * a for d in direction])
    for i in range(N_singles):
        direction = directions[np.random.randint(len(directions))]
        pos = np.random.uniform(high=(l_x, l_y))
        burg.append([x for x in pos] + [d * a for d in direction])
    return burg


def associate_color(b):
    burgers = list(direction_colors.keys())
    norm = np.sqrt(2) / np.linalg.norm(b) if np.linalg.norm(b) > 1e-3 else 0
    dist = []
    for burg in burgers:
        dist.append(np.linalg.norm(np.array(burg) - norm * b))
    return direction_colors[burgers[np.argmin(dist)]]


def bench_mark():
    N, rhoH = 900, 0.8

    def label(aglorithm, ic):
        alg_lbl = 'Metropolis' if algorithm == 'LMC' else 'Event Chain MC'
        ic_lbl = 'square' if initial_conditions == 'AF_square' else 'honeycomb'
        return alg_lbl + ', ic=' + ic_lbl

    def plot_str(algorithm, ic):
        s = '--' if algorithm == 'ECMC' else '-'
        s += 'b' if ic == 'AF_square' else 'g'
        return s

    # Figure 1: convergence of psi_mean
    plt.rcParams.update(params)
    plt.rcParams.update({'figure.figsize': (10, 10), 'legend.fontsize': size * 0.6})
    plt.figure()
    default_plt_kwargs['linewidth'] = 5
    for algorithm in ['LMC', 'ECMC']:
        for initial_conditions in ['AF_square', 'AF_triangle']:
            fname = join(
                op_path(rhoH, specif_op='psi_14', N=N, algorithm=algorithm, initial_conditions=initial_conditions),
                'mean_vs_real.txt')
            reals, psi_avg = np.loadtxt(fname, dtype=complex, unpack=True, usecols=(0, 1))
            reals = np.real(reals)
            psi_avg = np.abs(psi_avg)
            elapsed_hours = 24 * 2  # (len(psi_avg) - 1) / 2.0  # in hours
            plt.plot(reals * elapsed_hours / reals[-1], psi_avg, plot_str(algorithm, initial_conditions),
                     label=label(algorithm, initial_conditions), **default_plt_kwargs)
    plt.legend(loc=1)
    plt.xlabel('elapsed real time [h]')
    plt.ylabel('$|\\overline{\\psi_{4}}|$')
    # plt.ylim([0, 1.4])
    plt.grid()
    plt.savefig('graphs/bench_mark_psi_convergence')

    # Figure 2: histogram of |psi_4|
    plt.figure()
    for algorithm in ['LMC', 'ECMC']:
        for initial_conditions in ['AF_square', 'AF_triangle']:
            kwargs = {'N': N, 'algorithm': algorithm, 'initial_conditions': initial_conditions}
            hists = []
            op_dir = op_path(rhoH, specif_op='psi_14', **kwargs)
            files, reals = sort_prefix(op_dir, 'vec_')
            for k in range(60):
                psi = np.loadtxt(join(op_dir, files[k]), dtype=complex)
                hist, bin_edges = np.histogram(np.abs(psi), 50, density=True)
                hists.append(hist)
            hist = np.mean(hists, 0)
            plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist, plot_str(algorithm, initial_conditions),
                     label=label(algorithm, initial_conditions), **default_plt_kwargs)
    plt.legend(loc=2)
    plt.ylabel('PDF')
    plt.xlabel('$|\\psi_{4}|$')
    plt.grid()
    plt.savefig('graphs/bench_mark_psi_histograms')

    # Figure 3: histogram of z
    plt.figure()
    for algorithm in ['LMC', 'ECMC']:
        for initial_conditions in ['AF_square', 'AF_triangle']:
            kwargs = {'N': N, 'algorithm': algorithm, 'initial_conditions': initial_conditions}
            sim_path = join(sims_dir, sim_name(rhoH, **kwargs))
            write_or_load = WriteOrLoad(sim_path)
            hists = []
            for k in range(60):
                sp = write_or_load.last_spheres(k)[0]
                z = sp[:, 2]
                z = (z - np.mean(z)) * 2 / (np.max(z) - np.min(z))
                hist, bin_edges = np.histogram(z, 50, density=True)
                hists.append(hist)
            hist = np.mean(hists, 0)
            plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist, plot_str(algorithm, initial_conditions),
                     label=label(algorithm, initial_conditions), **default_plt_kwargs)
    plt.legend(loc=9)
    plt.ylabel('PDF')
    plt.xlabel('normalized sphere height')
    plt.grid()
    plt.savefig('graphs/bench_mark_z_histograms')


if __name__ == "__main__":
    """
    To recreate paper & supplement graphs, run this code after downloading the data and changing the variable
    'sims_dir' above to the appropriate path. Some of the figures, such as Fig. 1 In the paper, where created using a
    power point presentation, and the codes do not recreate them. Do feel free to contact us: ligolas@gmail.com.
    """

    rhoH_tetratic = 0.81
    realization = 94363239

    """ Fig. 2 - correlations """
    plot_pos_and_orientation([0.85, 0.84, 0.83, 0.82, rhoH_tetratic], [rhoH_tetratic, 0.8, 0.79, 0.78, 0.77])

    """ Fig. 3 - Plot global burger field before and after cleaning """
    x_lim_zoom_in, y_lim_zoom_in = [120, 147], [30, 46]
    coarse_grain_kwargs = {'xlim': [0, 260], 'ylim': [0, 260], 'distance_threshold': 10}  # 10 = 5 * sigma
    coarse_grain_burgers(rhoH_tetratic, realization=realization, plot_for_paper=True,
                         box=[x_lim_zoom_in, y_lim_zoom_in], **coarse_grain_kwargs)
    quiver_burger(rhoH_tetratic, x_lim_zoom_in, y_lim_zoom_in, realization=realization, given_color='gray')
    quiver_burger(rhoH_tetratic, x_lim_zoom_in, y_lim_zoom_in, bonds=False, plot_centers=False, frustrated_bonds=False,
                  realization=realization, quiv_surfix='_paired-4_clustered-D=10', new_fig=False)

    """ Fig 4 and Fig 5 Magnetic Structure"""
    plot_color_bargg_peaks([0.75, 0.78, rhoH_tetratic, 0.85], bragg_type="Bragg_Sm")

    """ Fig 5 Ising part"""
    plot_ising(plot_heat_capcity=False)


    """
    Supplement - Figs. 2,4-11,14-19. The rest of the codes need some adaption in order to recreate the graphs.
    The ECMC vs analytic solution is completely not here, but in a different matlab repository that can be found in my
    github user (DundeeB).
    """
    plot_psi_convergence(np.unique([0.83, rhoH_tetratic, 0.8, 0.78]))
    bench_mark()
    plot_z_histogram([0.71, 0.78, rhoH_tetratic, 0.85])
    for bragg_type in ['Bragg_S', 'Bragg_Sm']:
        plot_color_bargg_peaks([0.75, 0.78, rhoH_tetratic, 0.85], bragg_type=bragg_type)
    plot_magnetic_corr([0.85, 0.84, 0.83, rhoH_tetratic, 0.77])
    plot_local_psi_hist(np.unique([rhoH_tetratic, 0.8, 0.79, 0.785, 0.78, 0.775, 0.77])[-1::-1])

    """
    In the following Burgers field coarse graining graphs, the color scheme is different then in some of the graphs in
    the supplement. The color scheme used there is a 'no color scheme' version, where colors represent different
    clusters. Later on we changed the color scheme to correspond to the total Burgers field charge (see
    'associate_color' function)
    """
    coarse_grain_burgers(rhoH_tetratic, realization=realization, **coarse_grain_kwargs)
    coarse_grain_burgers(rhoH_tetratic, realization=92347176, **coarse_grain_kwargs)
    coarse_grain_burgers(rhoH_tetratic, realization=47424146, plot_for_paper=True, **coarse_grain_kwargs)
    for rho, real in zip([0.83, 0.84], [94861187, 95524231]):
        """
        rho=0.85 gives empty cleaned Burgers field, so the plotting code does not work.
        For rho=0.82 use realization 94818416
        """
        coarse_grain_burgers(rho, realization=real, plot_for_paper=True, **coarse_grain_kwargs)
    coarse_grain_null_model(ref_rhoH=rhoH_tetratic, ref_real=realization, distance_threshold=10)

    # """Used for extra test we performed later, graphs does not exist in paper or supplemental:"""
    # gradients_around_optimal_k([0.75, 0.78, 0.81, 0.85])
    # plot_psi_convergence([0.75, 0.85, 0.78, 0.81], orientation=True)

    print('Finished succesfully!')
    plt.show()
