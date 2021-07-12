from create_all_graphs import *
import sklearn
from numpy import unique
from numpy import where
from sklearn.cluster import *
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot
from SnapShot import WriteOrLoad


# I first try out all algorithms from https://machinelearningmastery.com/clustering-algorithms-with-python/

def plot_clustering(X, algorithm, *args, **kwargs):
    # define dataset
    # X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
    #                            random_state=4)
    # define the model
    model = algorithm(*args, **kwargs)
    # fit the model
    try:
        # if 'affinity' in kwargs and kwargs['affinity'] == 'precomputed':
        #     model.
        model.fit(X)
        # assign a cluster to each example
        yhat = model.predict(X)
    except AttributeError:
        yhat = model.fit_predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()


rhoH = 0.8
op_dir = op_path(rhoH, 'burger_vectors')
realization = 64155333
# clean_name = join(op_dir, 'vec_' + str(realization) + '.txt')
# burg = np.loadtxt(clean_name)/2
clean_name = join(op_dir, 'vec_' + str(realization) + '_pair-cleaned-3-times.txt')
burg = np.loadtxt(clean_name)
write_or_load = WriteOrLoad(join(sims_dir, sim_name(rhoH)))
l_x, l_y, _, _, _, _, _, _ = write_or_load.load_Input()
cyc = lambda p1, p2: cyc_dist(p1, p2, [l_x, l_y])

# plot_clustering(burg[:, :2], AffinityPropagation, damping=0.7)
# plot_clustering(burg[:, :2], Birch, threshold=0.01, n_clusters=1000)
# plot_clustering(burg[:, :2], DBSCAN, eps=3, min_samples=9)
# plot_clustering(burg[:, :2], KMeans, n_clusters=1000)  # seem to work pretty well
# plot_clustering(burg[:, :2], AgglomerativeClustering, n_clusters=1000)
plot_clustering(burg[:, :2], AgglomerativeClustering, distance_threshold=3, compute_full_tree=True, n_clusters=None,
                linkage='single', affinity=lambda X: pairwise_distances(X, metric=cyc))
                # linkage='single', affinity='precomputed', metric=cyc)

plt.quiver(burg[:, 0], burg[:, 1], burg[:, 2], burg[:, 3], angles='xy', scale_units='xy', scale=1,
           label='Burger field', width=3e-3, zorder=7)  # headwidth=3)  # , headlength=10, headaxislength=6
plt.axis('equal')
