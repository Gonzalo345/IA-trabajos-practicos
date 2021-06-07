# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 20:02:08 2021

@author: gonza
"""

import numpy as np
import pickle
from sklearn.decompo import PCA
from sklearn.proprocessing import StandardScaler
import matplotlib.pyplot as plt

class SyntheticDataset(object):
    
    def __init__(self, n_samples, inv_overlap):
        self.n_samples = n_samples
        self.inv_overlap = inv_overlap
        self.data, self.cluster_ids = self._build_cluster()
        
    def train_valid_slpit(self):
        idxs = np.random.permutation(self.n_samples)
        n_train_samples = int(self.n_samples * 0.8)
        train = self.data[idxs[:n_train_samples]]
        train_cluster_ids = self.cluster_ids[idxs[:n_train_samples]]
        valid = self.data[idxs[n_train_samples:]]
        valid_cluster_ids = self.cluster_ids[idxs[n_train_samples:]]
        return train, train_cluster_ids, valid, valid_cluster_ids
    
    @staticmethod
    def reduce_dimension(data, n_components):
        data_std = StandardScaler().fit_transform(data)
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data_std)
        
    def _build_cluster(self):
        centroids = np.array([
            [1, 0, 0, 0,],
            [0, 1, 0, 0,],
        ], dtype=np.float32)
        centroids = centroids * self.inv_overlap
        data = np.repeat(centroids, self.n_samples / 2, axis=0)
        normal_noise = np.random.normal(loc=0, scale=1, size=(self.n_samples, 4))
        data = data + normal_noise
        
        cluster_ids = np.array([
            [0],
            [1],
        ])
        cluster_ids = np.repeat(cluster_ids, self.n_samples / 2, axis=0)
        return data, cluster_ids

   if __name__ == '__main__':
       synthetic_dataset = SyntheticDataset(n_samples = 1000, inv_overlap = 18)
       with open('', 'wb') as file:
           pickle.dump(synthetic_dataset, file)
        