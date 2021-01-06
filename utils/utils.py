import numpy as np
import torch
from torch import nn

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


################################################################################
# 双线性解码器
class BiLinearDecoder(nn.Module):
    def __init__(self, hid_dim, out_dim, num_basis=2, dropout=0.0):
        super(BiLinearDecoder, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.dropout = nn.Dropout(dropout)
        self.Ps = nn.ParameterList(
          nn.Parameter(torch.randn(self.hid_dim, self.hid_dim))
          for _ in range(self.num_basis))
        self.combine_basis = nn.Linear(self.num_basis, self.out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # forward
    def forward(self, source_feat, dest_feat):
        source_feat = self.dropout(source_feat)
        dest_feat = self.dropout(dest_feat)
        basis_out = []
        for i in range(self.num_basis):
            source_feat = source_feat @ self.Ps[i]
            ## 两个张量各行一一点乘
            sr = torch.mul(source_feat, dest_feat).sum(dim=1).view(-1,1)
            basis_out.append(sr)
        out = torch.cat(basis_out, dim=1)
        out = self.combine_basis(out)
        return out
###############################################################################


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:

            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


#####################################################################################
### 更改 Neighbour Finder
def get_neighbor_finder(data, uniform, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp, label in zip(data.sources, data.destinations,
                                                               data.edge_idxs,
                                                               data.timestamps,
                                                               data.labels):
        adj_list[source].append((destination, edge_idx, timestamp, label))
        adj_list[destination].append((source, edge_idx, timestamp, label))

    return NeighborFinder(adj_list)


class NeighborFinder:
    def __init__(self, adj_list, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        self.node_to_edge_labels = []

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp, label)
            # We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))
            self.node_to_edge_labels.append(np.array([x[3] for x in sorted_neighhbors]))

        # self.uniform = uniform

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time, n_relations=5):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 4 lists: neighbors, edge_idxs, timestamps

        """
        out_neighbours = []
        out_edge_idx = []
        out_timestamp = []
        for i in range(n_relations):
            idx = np.where((self.node_to_edge_timestamps[src_idx] < cut_time) &
                           (self.node_to_edge_labels[src_idx] == i+1))
            out_neighbours.append(self.node_to_neighbors[src_idx][idx])
            out_edge_idx.append(self.node_to_edge_idxs[src_idx][idx])
            out_timestamp.append(self.node_to_edge_timestamps[src_idx][idx])
        #i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return out_neighbours, out_edge_idx, out_timestamp

    ################################################################################
    ### 此处更改
    def get_temporal_neighbor(self, source_nodes, timestamps, n_relations=5, n_neighbors=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        ### each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i]
        ### with an interaction happening before cut_time_l[i]
        neighbors = np.zeros((len(source_nodes), n_relations, tmp_n_neighbors)).astype(np.int32)
        edge_times = np.zeros((len(source_nodes), n_relations, tmp_n_neighbors)).astype(np.float32)
        edge_idxs = np.zeros((len(source_nodes), n_relations, tmp_n_neighbors)).astype(np.float32)

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                                                     timestamp,
                                                                                     n_relations)
            for j in range(n_relations):
                if len(source_neighbors[j]) > 0 and n_neighbors > 0:
                    # Take most recent interactions
                    source_edge_times_j = source_edge_times[j][-n_neighbors:]
                    source_neighbors_j = source_neighbors[j][-n_neighbors:]
                    source_edge_idxs_j = source_edge_idxs[j][-n_neighbors:]

                    assert (len(source_neighbors_j) <= n_neighbors)
                    assert (len(source_edge_times_j) <= n_neighbors)
                    assert (len(source_edge_idxs_j) <= n_neighbors)

                    neighbors[i, j, n_neighbors - len(source_neighbors_j):] = source_neighbors_j
                    edge_times[i, j, n_neighbors - len(source_edge_times_j):] = source_edge_times_j
                    edge_idxs[i, j, n_neighbors - len(source_edge_idxs_j):] = source_edge_idxs_j

        return neighbors, edge_idxs, edge_times

