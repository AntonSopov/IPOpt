import numpy as np
from Problem import *
import Global as Gl

class Clustering:
    _cluster_assignment = None      # np.array(1d, int), assignment of each data element
    _total_clusters = None          # int, total number of clusters
    _centre = None                  # np.array(2d, float), cluster centres
    _member_counter = None          # np.array(1d, int), counter of cluster members

    def __init__(self, locus_encoding):
        # initializing
        self._cluster_assignment = np.ones(Gl.problem.ndata(), dtype = int) * (-1)
        self._total_clusters = 0
        previous = np.empty(Gl.problem.ndata(), dtype = int)

        # getting assinment of each data element to a cluster 
        # (extracting connected components of the graph)
        for i in range(Gl.problem.ndata()):
            ctr = 0

            # if the element is unassigned, assign cluster
            if (self._cluster_assignment[i] == -1):

                # assign to cluster, identify neighbour
                self._cluster_assignment[i] == self._total_clusters
                previous[ctr] = i
                neighbour = locus_encoding[i]
                ctr += 1

                # assigning every element in the component to the graph
                while self._cluster_assignment[neighbour] == -1:
                    self._cluster_assignment[neighbour] = self._total_clusters
                    previous[ctr] = neighbour
                    neighbour = locus_encoding[neighbour]
                    ctr += 1
                
                # if a previously assigned neighbour is reached and it is assigned to a different 
                # cluster X,then it's actually a same component and all elements in previous list 
                # should be re-assigned to X
                if self._cluster_assignment[neighbour] != self._total_clusters:
                    ctr -= 1
                    while ctr >= 0:
                        self._cluster_assignment[previous[ctr]] = self._cluster_assignment[neighbour]
                        ctr -= 1
                else:
                    self._total_clusters += 1
                
        self.compute_cluster_centres()


    def compute_cluster_centres(self):
        # initialization
        self._centre = np.zeros((self._total_clusters, Gl.problem.mdim()), dtype = float)
        self._member_counter = np.zeros(self._total_clusters, dtype = int)

        # average data elements of each cluster
        for i in range(Gl.problem.ndata()):
            self._centre[self._cluster_assignment[i]] += Gl.problem.element(i)
            self._member_counter[self._cluster_assignment[i]] += 1
        
        # dividing clusters by its member count
        for i in range(self._total_clusters):
            if self._member_counter[i] > 0:
                self._centre[i] /= self._member_counter[i]
    
    def update_cluster_centres(self):
        # initialization
        self._centre = np.zeros((self._total_clusters, Gl.problem.mdim()), dtype = float)

        # average data elements of each cluster
        for i in range(Gl.problem.ndata()):
            self._centre[self._cluster_assignment[i]] += Gl.problem.element(i)
        
        # dividing clusters by its member count
        for i in range(self._total_clusters):
            if self._member_counter[i] > 0:
                self._centre[i] /= self._member_counter[i]
    
    # changes the assignment of a data element
    def update_assignment(self, element, cluster):
        # decreasing member counter of original cluster
        self._member_counter[self._cluster_assignment[element]] -= 1
        
        # updating assignment
        self._cluster_assignment[element] = cluster

        # increasing member counter of new cluster
        self._member_counter[cluster] += 1

    
    
    # getter functions
    def assignment(self, i): return self._cluster_assignment[i]
    def total_clusters(self): return self._total_clusters
    def centre(self, i): return self._centre[i]
    def member_counter(self, i): return self._member_counter[i]