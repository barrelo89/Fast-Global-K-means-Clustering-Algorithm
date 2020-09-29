import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class GKMClustering():

    def __init__(self, n_cluster):
        self.n_cluster = n_cluster

    def NearestCentroid(self, data_point, data_array):

        squared_distance_mtx = ((data_array - data_point)**2).sum(axis = 1)
        min_idx = np.argmin(squared_distance_mtx)
        return min_idx, squared_distance_mtx[min_idx]

    def centroid_candidate(self, input_data, centroids_array):

        objective_value_list = []

        for data_sample in input_data:

            min_idx, min_squared_distance_ctr = self.NearestCentroid(data_sample, centroids_array)
            squared_distance_mtx = ((input_data - data_sample)**2).sum(axis = 1)
            objective_value = np.maximum(min_squared_distance_ctr - squared_distance_mtx, 0).sum()
            objective_value_list.append(objective_value)

        candidate_idx = np.argmax(objective_value_list)

        return candidate_idx, input_data[candidate_idx]

    def KMClustering(self, input_data, initial_centroids_array):

        candidate_idx, candidate = self.centroid_candidate(input_data, initial_centroids_array)

        initial_centroids_array = np.vstack((initial_centroids_array, candidate))

        n_cluster, dim = initial_centroids_array.shape

        km = KMeans(n_clusters = n_cluster, init = initial_centroids_array).fit(input_data)

        return km

    def GlobalKMClustering(self, input_data):

        num_sample, dim = input_data.shape
        initial_centroid = input_data[np.random.choice(num_sample)] #centroid for K = 1

        if initial_centroid.ndim == 1:
            initial_centroid = np.expand_dims(initial_centroid, axis = 0)

        n_cluster, n_dim = initial_centroid.shape

        km = KMeans(n_clusters = n_cluster, init = initial_centroid).fit(input_data)
        updated_centroids = km.cluster_centers_

        for n_cluster in range(1, self.n_cluster):

            km = self.KMClustering(input_data, updated_centroids)
            updated_centroids = km.cluster_centers_

        return km

'''
HOW TO USE THE FAST GLOBAL K MEANS CLUSTERING
data = np.random.randint(low = 0, high = 50, size = 200).reshape(100, 2)

GKMC = GKMClustering(n_cluster = 3)
km = GKMC.GlobalKMClustering(data)

print(km.cluster_centers_)
print(km.labels_)

plt.scatter(data[km.labels_ == 0, 0], data[km.labels_ == 0, 1], color = 'r')
plt.scatter(data[km.labels_ == 1, 0], data[km.labels_ == 1, 1], color = 'g')
plt.scatter(data[km.labels_ == 2, 0], data[km.labels_ == 2, 1], color = 'b')
plt.tight_layout()
plt.show()
'''






























#end
