import matplotlib.pyplot as plt
import numpy as np
import os, re
from sys import path as syspath

syspath.append('../OOP_hard_sphere_event_chain/')
from post_process import Ising

sims_dir = 'C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\post_process\\from_ATLAS3.0'
default_plt_kwargs = {'linewidth': 5, 'markersize': 10}
size = 30
params = {'legend.fontsize': size * 0.75, 'figure.figsize': (10, 10), 'axes.labelsize': size, 'axes.titlesize': size,
          'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75}
plt.rcParams.update(params)


def join(*args, **kwargs):
    return os.path.join(*args, **kwargs)


def sim_name(rhoH, N=int(9e4), h=0.8, initial_conditions='AF_square'):
    return 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rhoH) + '_' + initial_conditions + '_ECMC'


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
        for mn in ['14', '23', '16']:
            lbl = re.sub('psi ' + mn, '$g_{' + mn + '}$', lbl)
    if lbl.startswith('Bragg Sm'):
        lbl = re.sub('Bragg Sm', '$g_k^M$', lbl)
    if lbl.startswith('Bragg S'):
        lbl = re.sub('Bragg S', '$g_k$', lbl)
    for N, N_ in zip(['10000', '40000', '90000'], ['1e4', '4e4', '9e4']):
        lbl = re.sub(N, N_, lbl)
    return lbl


def plot_corr(rhoH, specific_op, realizations=1, poly_slope=None, *args, **kwargs):
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
        plt.loglog(x, y, label=prepare_lbl('rhoH=' + str(rho)), **default_plt_kwargs)
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
        plt.loglog(x, y, '--k', label='$x^{1/4\ or\ 1/3}$', **default_plt_kwargs)
    plt.grid()
    plt.legend(loc=4)
    return


def quiver_burger(rhoH, xlim, ylim, realization=None, *args, **kwargs):
    plt.figure()
    # op_dir = op_path(rhoH, 'burger_vectors', *args, **kwargs)
    op_dir = op_path(rhoH, 'burger_vectors_orientation_rad=10.0', *args, **kwargs)
    sim = join(sims_dir, sim_name(rhoH, *args, **kwargs))
    if realization is None:
        files, reals = sort_prefix(op_dir, 'vec_')
        realization = reals[0]
        file = files[0]
    else:
        file = 'vec_' + str(realization) + '.txt'
    op = Ising(sim_path=sim, k_nearest_neighbors=4, directed=False, centers=np.loadtxt(join(sim, str(realization))),
               spheres_ind=realization)
    op.initialize(random_initialization=False, J=-1)
    graph = op.graph
    spins = op.z_spins
    x = op.spheres[:, 0]
    y = op.spheres[:, 1]
    z = op.spheres[:, 2]
    up = np.array(
        [(z_ > np.mean(z)) and xlim[0] < x_ < xlim[1] and ylim[0] < y_ < ylim[1] for (x_, y_, z_) in zip(x, y, z)])
    down = np.array(
        [(z_ <= np.mean(z)) and xlim[0] < x_ < xlim[1] and ylim[0] < y_ < ylim[1] for (x_, y_, z_) in zip(x, y, z)])
    for i in range(len(x)):
        for j in graph.getrow(i).indices:
            if j > i or not (up[i] or down[i]) or not (up[j] or down[j]):
                continue
            ex = [x[i], x[j]]
            ey = [y[i], y[j]]
            if (ex[1] - ex[0]) ** 2 + (ey[1] - ey[0]) ** 2 > 10 ** 2:
                continue
            if spins[i] * spins[j] > 0:
                plt.plot(ex, ey, 'r')
            if spins[i] * spins[j] < 0:
                plt.plot(ex, ey, color='lightgray')
    plt.plot(x[up], y[up], '.r', label='up', markersize=6)
    plt.plot(x[down], y[down], '.b', label='down', markersize=6)
    plt.axis('equal')
    # plt.legend()
    burg = np.loadtxt(join(op_dir, file))
    I_box = np.array([xlim[0] < x_ < xlim[1] and ylim[0] < y_ < ylim[1] for (x_, y_) in zip(burg[:, 0], burg[:, 1])])
    burg = burg[I_box, :]
    plt.quiver(burg[:, 0], burg[:, 1], burg[:, 2], burg[:, 3], angles='xy', scale_units='xy', scale=1,
               label='Burger field')

    plt.savefig('graphs/burger_vectors')
    # TODO: add burger vectors
    return


def plot_pos_and_orientation():
    plt.figure()
    rhos = [0.85, 0.83, 0.8, 0.77]
    corr_ylim = [1e-2, 1]
    corr_xlim = [0.8, 1e2]

    plt.subplot(211)
    plot_corr(rhos, 'psi_14', poly_slope=0.25)
    plt.ylim(corr_ylim)
    plt.xlim(corr_xlim)
    plt.ylabel(prepare_lbl('psi_14'))

    plt.subplot(212)
    plot_corr(rhos[:-1], 'Bragg_S', poly_slope=1.0 / 3)
    plt.ylim(corr_ylim)
    plt.xlim(corr_xlim)
    ax = plt.gca()
    ax.legend_ = None
    plt.xlabel('$\Delta$r/$\sigma$')
    plt.ylabel(prepare_lbl('Bragg_S'))

    plt.savefig('graphs/orientation_and_position_corr')


if __name__ == "__main__":
    # plot_pos_and_orientation()
    quiver_burger(0.8, [250, 310], [230, 290], realization=47806807)
    # add git, github ext

plt.show()
