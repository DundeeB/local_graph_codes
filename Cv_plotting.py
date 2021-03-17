from create_all_graphs import sim_name, op_path, join, sort_prefix, sims_dir
import numpy as np
import matplotlib.pyplot as plt
from sys import path as syspath

syspath.append('C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\OOP_hard_sphere_event_chain')
from post_process import Ising

size = 30
params = {'legend.fontsize': size * 0.75, 'figure.figsize': (15, 10), 'axes.labelsize': 0.85 * size,
          'axes.titlesize': size, 'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75}
plt.rcParams.update(params)

for rhoH in [0.85, 0.8, 0.75]:
    for N in [9e4]:
        try:
            sim = join(sims_dir, sim_name(rhoH, N=int(N)))
            op_dir = op_path(rhoH, specif_op='Ising_k=4_undirected', N=N)
            files, reals = sort_prefix(op_dir, 'Cv_vs_J')
            Cvfile, realization = files[0], reals[0]
            Cv = np.loadtxt(join(op_dir, Cvfile))
            # plt.subplot(211)
            J = Cv[:-1, 0] + np.diff(Cv[:, 0]) / 2
            # ising_op = Ising(sim_path=sim, k_nearest_neighbors=4, directed=False,
            #                  centers=np.loadtxt(join(sim, str(realization))),
            #                  spheres_ind=realization)
            # z = ising_op.bonds_num/ising_op.N
            z = 4
            Cv_dfdJ = np.diff(Cv[:, 2]) / np.diff(Cv[:, 0]) * J ** 2 * 2 * z
            plt.plot(J, Cv_dfdJ, '.-',
                     label='$\\rho_H$=' + str(rhoH) + ', N=' + str(int(N)))
            # plt.plot(Cv[:, 0], Cv[:, 2], label='$\\rho_H$=' + str(rhoH) + ', N=' + str(int(N)))
            # plt.subplot(212)
            # plt.plot(Cv[:, 0], Cv[:, 1], '.-')
        except Exception:
            print(Exception)
# plt.subplot(211)
plt.ylabel('Cv from d(frustration)/dJ')
plt.grid()
plt.legend()
# plt.subplot(212)
# plt.ylabel('Cv from var(E)')
# plt.xlabel('J')
plt.grid()
plt.show()
