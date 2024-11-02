import numpy as np
import logging

logger = logging.getLogger("renormalizer")

def modes_filter(w, lamb, lamb_threshold, nac=None, nac_threshold=None):

    if isinstance(lamb_threshold, int):
        # the modes with largest lambda 
        index_lamb = np.argsort(lamb, axis=None)[::-1][:lamb_threshold]
    else:
        # the modes with lambda larger than lamb_threshold
        index_lamb = np.where(np.array(lamb) >  lamb_threshold)[0]
    
    if nac is not None:
        nac2 = np.abs(nac)**2*w
        nac2_sum = np.sum(nac2)
        nac_sort = np.argsort(nac2, axis=None)[::-1]
        assert nac_threshold > 0 and nac_threshold < 1 # percent
        tot = 0
        for idx in range(len(w)):
            tot += nac2[nac_sort[idx]]
            if tot > nac2_sum * nac_threshold:
                break
        index_nac = nac_sort[:idx+1]
    
    if nac is not None:
        index = np.array(list(set(index_lamb)| set(index_nac)))
        nac_new = nac[index]
    else:
        index = index_lamb
        nac_new = None

    w_new = w[index]
    lamb_new = np.zeros_like(w_new)
    
    portion = np.sum(lamb[index]) / np.sum(lamb)
    logger.info(f"the portion of selected reorganization energy is {portion}")

    for idx in range(len(w)):
        nearest_idx = np.abs(w[idx]-w_new).argmin()
        lamb_new[nearest_idx] += lamb[idx]

    return w_new, lamb_new, nac_new

#def kmeans_clustering():
#from sklearn.cluster import KMeans  #For applying KMeans
#nclusters = 8
#kmeans = KMeans(n_clusters=nclusters, n_init=10, random_state=0, max_iter=1000)
#wt_kmeansclus = kmeans.fit(w0.reshape(-1,1), sample_weight=lamb0)
#predicted_kmeans = kmeans.predict(w0.reshape(-1,1), sample_weight = lamb0)
#print(predicted_kmeans)
#print(wt_kmeansclus.cluster_centers_*au2cm)

#lamb_new = np.zeros(nclusters)
#for idx, icluster in enumerate(predicted_kmeans):
#    lamb_new[icluster] += lamb0[idx]
#print(lamb_new*au2cm)

