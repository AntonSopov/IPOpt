import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances

class Problem:
    # * - input parameter
    # ^ - optional input parameter
    _data = None                #  np.array(2d, float), data points
    _filename = None            #* string, data filename
    _ndata = None               #  int, number of data elements
    _mdim = None                #  int, number of dimensions

    _is_label = None            #  bool, flag if labels are provided
    _label = None               #  np.array(1d, int), labels (if labels provided)
    _num_real_clusters = None	#  int, number of real cluster (if labels provided)
    
    _separator = None           #  separator symbol between values in file
    _normalize = None           #* bool, flag if normalising data
    _distance = None            #  string, distance measure

    # precomputed info
    _distance_matrix = None     #  np.array(2d, float), pairwise-distances between data elements
    _nearest_neighbours = None  #  np.array(2d, int), nearest neighbour matrix
    _neighbour_rank = None      #  np.array(2d, int), neighbour rank matrix (positions in nearest neighbour list)
    
    _priority_edges = None      #  [(int, float)], sorted list of (idx, interesingness) of interesting edges
    _num_priority_edges = None  #  int, number of interesting edges found

    # for mst calculation
    _mst = None                 #  np.array(1d, int)
    


    # [ndata, mdim, islabel, realclusters]
    def __init__(self, filename, data_info, settings):
        # filename
        self._filename = filename

        # loading data info from "settings" dict
        self._ndata = data_info['ndata']
        self._mdim = data_info['mdim']
        self._is_label = data_info['is_label']
        self._num_real_clusters = data_info['num_real_clusters']

        # settings
        self._distance = settings['distance']
        self._normalize = settings['normalize']
        self._separator = settings['separator']

        self.configure()
    
    def configure(self):
        # data preparation
        print('data_preparation')
        self.data_preparation()

        self.compute_distance_matrix()

        self.compute_nearest_neighbours()

        self.compute_mst()
    
    def data_preparation(self):
        # loading data and labels
        if not self._is_label:
            self._data = pd.read_csv(self._filename, sep = self._separator).to_numpy() # , header = None
        else:
            self._data = pd.read_csv(self._filename, sep = self._separator, usecols = np.arange(self._mdim)).to_numpy()
            self._label = pd.read_csv(self._filename, sep = self._separator, usecols = [self._mdim]).to_numpy().reshape((self._ndata))
        
        # normalizing data
        scaler = MinMaxScaler().fit(self._data)
        self._data = scaler.transform(self._data)
    
    def compute_distance_matrix(self):
        # calculating distances
        dist = pairwise_distances(self._data, metric = self._distance)

        # normalizing, extracting the lower triangular matrix
        self._distance_matrix = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
        # self._distance_matrix = np.tril( (dist - np.min(dist)) / (np.max(dist) - np.min(dist)) )
        
    def compute_nearest_neighbours(self):
        # initializing arrays
        self._nearest_neighbours = np.empty((self._ndata, self._ndata), dtype = int)
        self._neighbour_rank = np.empty((self._ndata, self._ndata), dtype = int)
        idx_tuple = np.empty(self._ndata - 1, dtype = int)
        val_tuple = np.empty(self._ndata - 1)

        for i in range(self._ndata):
            cur = 0
            # getting index and values for ith element
            for j in range(self._ndata):
                if i != j:
                    idx_tuple[cur] = j
                    val_tuple[cur] = self._distance_matrix[i][j]
                    cur += 1

            # sorting tuples
            idx_tuple = idx_tuple[np.argsort(val_tuple)].copy()
            val_tuple = np.sort(val_tuple).copy()

            # saving nn and ranks
            for j in range(self._ndata - 1):
                self._nearest_neighbours[i][j + 1] = idx_tuple[j]
                self._neighbour_rank[i][idx_tuple[j]] = j + 1
            
            # i is the closest neighbour to i
            self._nearest_neighbours[i][0] = i
            self._neighbour_rank[i][i] = 0

    # precomputation of minimum spanning tree (MST) using Prim's algorithm
    def compute_mst(self):
        # initializing the mst
        self._mst = np.empty(self._ndata, dtype = int)

        self._num_priority_edges = 0
        _priority_edges_idx = np.empty(self._ndata, dtype = int)
        _priority_edges_val = np.empty(self._ndata, dtype = float)

        # marking all nodes (items) as unselected
        node = np.empty(self._ndata, dtype = int)
        total_nodes = 0
        selected = np.zeros(self._ndata, dtype = bool)

        # randomly selecting initial node (starting point)
        r = np.random.randint(0, self._ndata - 1)
        node[0] = r
        selected[r] = True
        total_nodes += 1

        # Prim's method main cycle
        while (total_nodes < self._ndata):
            # finding the shortest edge (n1, n2) between a connected node and an unconnected node
            possible_connections = self._distance_matrix[selected == True][:, selected == False]
            idx_remain = [np.arange(self._ndata)[selected == True], np.arange(self._ndata)[selected == False]]
            
            edge = np.unravel_index(np.argmin(possible_connections), (total_nodes, self._ndata - total_nodes))
            n1 = idx_remain[0][edge[0]]
            n2 = idx_remain[1][edge[1]]
            
            # saving new edge found
            self._mst[n2] = n1
            if total_nodes == 1:
                self._mst[n1] = n2
            
            # including node n2 and marking it as selected
            node[total_nodes] = n2
            total_nodes += 1
            selected[n2] = True


            # getting nn ranks
            mock_l = self._neighbour_rank[n1][n2]
            mock_k = self._neighbour_rank[n2][n1]

            # computing priority based on interestingness and distance
            _priority_edges_idx[self._num_priority_edges] = n2
            _priority_edges_val[self._num_priority_edges] = np.min([mock_l, mock_k]) + self._distance_matrix[n1][n2]
            self._num_priority_edges += 1

            # if there are 2 nodes, there is only 1 edge yet, its indices are n1 and n2
            if total_nodes == 2:
                _priority_edges_idx[self._num_priority_edges] = n1
                _priority_edges_val[self._num_priority_edges] = np.min([mock_l, mock_k]) + self._distance_matrix[n1][n2]
                self._num_priority_edges += 1
            
        # sorting edges in descending order of priority
        _priority_edges_idx = _priority_edges_idx[np.argsort(_priority_edges_val)].copy()
        _priority_edges_val = np.sort(_priority_edges_val).copy()

        # making priority tuples
        self._priority_edges = []
        for i in range(self._ndata):
            self._priority_edges.append( (_priority_edges_idx[i], _priority_edges_val[i]) )



    # getter functions
    def data(self): return self._data
    def element(self, i): return self._data[i]
    def ndata(self): return self._ndata
    def mdim(self): return self._mdim    
    def is_label(self): return self._is_label
    def label(self, i = None): return self._label if i == None else self._label[i]
    def num_real_clusters(self): return self._num_real_clusters
    def distance(self, i, j): return self._distance_matrix[i][j]
    def distance_measure(self): return self._distance
    def neighbour(self, i, j): return self._nearest_neighbours[i][j]
    def neighbour_rank(self, i, j): return self._neighbour_rank[i][j]
    def mst_edge(self, i): return self._mst[i]
    def priority_edge(self, i): return self._priority_edges[i][0]
    def num_priority_edges(self, i): return self._num_priority_edges