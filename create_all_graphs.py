import matplotlib.pyplot as plt
import numpy as np
import os, re
from sys import path as syspath

syspath.append('C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\OOP_hard_sphere_event_chain')
from post_process import Ising
from mpl_toolkits import mplot3d

sims_dir = 'C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\post_process\\from_ATLAS3.0'
default_plt_kwargs = {'linewidth': 5, 'markersize': 10}
size = 30
params = {'legend.fontsize': size * 0.75, 'figure.figsize': (10, 10), 'axes.labelsize': size, 'axes.titlesize': size,
          'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75}
plt.rcParams.update(params)


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


def plot_corr(rhoH, specific_op, realizations=1, poly_slope=None, single_slope_label=False, *args, **kwargs):
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
        lbl = '$x^{1/4\ or\ 1/3}$' if not single_slope_label else '$x^{' + str(np.round(poly_slope, 2)) + '}$'
        plt.loglog(x, y, '--k', label=lbl, **default_plt_kwargs)
    plt.grid()
    plt.legend(loc=4)
    return


def plot_pos_and_orientation(rhos):
    plt.rcParams.update({'figure.figsize': (10, 10)})
    plt.figure()
    corr_ylim = [1e-2, 1]
    corr_xlim = [0.8, 1e2]

    plt.subplot(211)
    plot_corr(rhos, 'psi_14', poly_slope=0.25, N=4e4)
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


def quiver_burger(rhoH, xlim, ylim, realization=None, bonds=False, *args, **kwargs):
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
                if spins[i] * spins[j] > 0:
                    plt.plot(ex, ey, 'r')
                if spins[i] * spins[j] < 0:
                    plt.plot(ex, ey, color='lightgray')
    plt.plot(x[up], y[up], '.r', label='up', markersize=10)
    plt.plot(x[down], y[down], '.b', label='down', markersize=10)
    plt.axis('equal')
    # plt.legend()
    burg = np.loadtxt(join(op_dir, file)) / 2
    I_box = np.array([xlim[0] < x_ < xlim[1] and ylim[0] < y_ < ylim[1] for (x_, y_) in zip(burg[:, 0], burg[:, 1])])
    burg = burg[I_box, :]
    plt.quiver(burg[:, 0], burg[:, 1], burg[:, 2], burg[:, 3], angles='xy', scale_units='xy', scale=1,
               label='Burger field', width=3e-3)  # headwidth=3)  # , headlength=10, headaxislength=6
    plt.savefig('graphs/burger_vectors')
    return


def plt_cv(rhoH, *args, **kwargs):
    default_plt_kwargs['linewidth'] = 3
    op_dir = op_path(rhoH, specif_op='Ising_k=4_undirected', *args, **kwargs)
    sim = join(sims_dir, sim_name(rhoH, *args, **kwargs))
    cvfiles, reals = sort_prefix(op_dir, 'Cv_vs_J')
    Cvs = []
    for file, realization in zip(cvfiles, reals):
        cv_mat = np.loadtxt(join(op_dir, file))
        ising_op = Ising(sim_path=sim, k_nearest_neighbors=4, directed=False,
                         centers=np.loadtxt(join(sim, str(realization))), spheres_ind=realization)
        z = ising_op.bonds_num / ising_op.N
        J = cv_mat[:-1, 0] + np.diff(cv_mat[:, 0]) / 2
        dfdJ = np.diff(cv_mat[:, 2]) / np.diff(cv_mat[:, 0])
        Cvs.append(2 * J ** 2 * z * dfdJ)

    min_length = min([len(Cv) for Cv in Cvs])
    Cv = np.mean([Cv[:min_length] for Cv in Cvs], 0)
    plt.plot(J, Cv, label=prepare_lbl('rhoH=' + str(rhoH)), **default_plt_kwargs)


def frustration(rhoH, *args, **kwargs):
    op_dir = op_path(rhoH, 'Graph', *args, **kwargs)
    spheres_frustration = np.loadtxt(join(op_dir, sort_prefix(op_dir, 'frustration_k=4_undirected_')[0][0]))

    op_dir = op_path(rhoH, 'Ising_k=4_undirected', *args, **kwargs)
    cv_file = sort_prefix(op_dir, 'Cv_vs_J_')[0][0]
    J, Cv, frustration = np.loadtxt(join(op_dir, cv_file), unpack=True, usecols=(0, 1, 2))

    anneal_file = sort_prefix(op_dir, 'anneal_')[0][0]
    A = np.loadtxt(os.path.join(op_dir, anneal_file))
    min_f = float('inf')
    reals = int((A.shape[1] - 1) / 2)
    for i in range(1, reals + 1):
        m = min(A[:, i])
        if m < min_f:
            min_f = m
    return min_f, spheres_frustration, frustration[np.argmax(Cv)],


def plot_ising(rhos, *args, **kwargs):
    plt.rcParams.update({'figure.figsize': (10, 10), 'axes.labelsize': 0.5 * size, 'xtick.labelsize': size * 0.5,
                         'ytick.labelsize': size * 0.5, 'legend.fontsize': size * 0.5})
    plt.figure()

    plt.subplot(211)
    for rhoH in rhos:
        plt_cv(rhoH, *args, **kwargs)
    plt.xlabel('$\\beta$J')
    plt.ylabel('$C_V$/$k_B$')
    plt.grid()
    plt.legend()

    plt.subplot(212)
    default_plt_kwargs['linewidth'] = 3
    N9e4dirs = [s for s in os.listdir(sims_dir) if s.startswith('N=9')]
    rhos = [float(re.split('(=|_)', dirname)[10]) for dirname in N9e4dirs]
    ground_state, spheres_frustration, max_cv_frustration = [], [], []
    for rho in rhos:
        a, b, c = frustration(rho)
        ground_state.append(a)
        spheres_frustration.append(b)
        max_cv_frustration.append(c)
    rhos = np.array(rhos)
    I = np.argsort(rhos)
    rhos = rhos[I]
    ground_state = np.array(ground_state)[I]
    spheres_frustration = np.array(spheres_frustration)[I]
    max_cv_frustration = np.array(max_cv_frustration)[I]
    plt.plot(rhos, ground_state, label='ground state', **default_plt_kwargs)
    plt.plot(rhos, spheres_frustration, label='spheres heights frustration', **default_plt_kwargs)
    plt.plot(rhos, max_cv_frustration, '.k', **default_plt_kwargs)
    p = np.polyfit(rhos, max_cv_frustration, 3)
    rhos_p = np.linspace(min(rhos), max(rhos))
    plt.plot(rhos_p, np.polyval(p, rhos_p), 'k', label='frustration at maximal $C_V$', **default_plt_kwargs)

    plt.legend()
    plt.grid()
    plt.ylabel('frustration')
    plt.xlabel(prepare_lbl('rhoH'))
    plt.savefig('graphs/Ising')
    return


def hist_local_psi(rhoH, rad=30, *args, **kwargs):
    default_plt_kwargs['linewidth'] = 3
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
    plt.plot(rho_avg, psi_avg, label=prepare_lbl('rhoH=' + str(rhoH)), **default_plt_kwargs)


def plot_local_psi_hist(rhos, rad=30, *args, **kwargs):
    plt.rcParams.update(params)
    plt.rcParams.update({'figure.figsize': (10, 10)})
    plt.figure()
    for rho in rhos:
        hist_local_psi(rho, rad, *args, **kwargs)
    plt.legend()
    plt.grid()
    plt.xlabel('|$\\psi_{14}^{r=' + str(rad) + '}$|')
    plt.ylabel('pdf')
    plt.savefig('graphs/local_psi_histogram')


def plot_magnetic_corr(rhos):
    plt.rcParams.update({'figure.figsize': (12, 10)})
    plt.figure()
    corr_ylim = [1e-2, 1]
    corr_xlim = [0.8, 1e2]

    plot_corr(rhos, 'Bragg_Sm', poly_slope=1.0 / 3, single_slope_label=True)
    plt.ylim(corr_ylim)
    plt.xlim(corr_xlim)
    plt.xlabel('$\Delta$r/$\sigma$')
    plt.ylabel(prepare_lbl('Bragg_Sm'))
    plt.savefig('graphs/magnetic_bragg_corr')


def plot_bragg_peak(rhoH, *args, **kwargs):
    plt.rcParams.update({'figure.figsize': (10, 10), 'axes.labelsize': 0.5 * size, 'xtick.labelsize': size * 0.5,
                         'ytick.labelsize': size * 0.5, 'legend.fontsize': size * 0.5})
    fig = plt.figure()
    axs = [fig.add_subplot(2, 1, 1, projection='3d'), fig.add_subplot(2, 1, 2, projection='3d')]
    for bragg_type, sub in zip(['Bragg_S', 'Bragg_Sm'], [0, 1]):
        try:
            op_dir = op_path(rhoH, bragg_type)
            data_file = sort_prefix(op_dir, 'vec_')[0][0]
            kx, ky, S_values = np.loadtxt(join(op_dir, data_file), unpack=True, usecols=(0, 1, 2))
            ax = axs[sub]
            ax.scatter(kx, ky, S_values, '.')
            if sub == 0:
                ax.set_title(prepare_lbl('rhoH=' + str(rhoH)))
                ax.set_zlabel('S(k)')
            else:
                ax.set_zlabel('$S^M(k)$')
            ax.set_xlabel('$k_x$')
            ax.set_ylabel('$k_y$')
        except Exception as err:
            print(err)
    plt.savefig('graphs/Bragg_peak')


def plot_annealing(rhoH, real=None, *args, **kwargs):
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


if __name__ == "__main__":
    # plot_pos_and_orientation([0.85, 0.83, 0.8, 0.77])
    # quiver_burger(0.8, [85, 106.7], [126, 140.1], realization=64155333, bonds=True)
    # plot_ising([0.85, 0.8, 0.75])
    # plot_local_psi_hist([0.77, 0.78, 0.8])

    # plot_local_psi_hist([0.77, 0.775, 0.78, 0.785, 0.8])
    # plot_magnetic_corr([0.85, 0.83, 0.8, 0.77])
    # plot_bragg_peak(0.8)
    plot_annealing(0.8, real=45738368)

plt.show()
