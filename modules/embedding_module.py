import torch
from torch import nn
import numpy as np
import math

from model.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        # self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        pass


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                             neighbor_finder, time_encoder, n_layers,
                                             n_node_features, n_edge_features, n_time_features,
                                             embedding_dimension, device, dropout)

        self.use_memory = use_memory
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, n_relations=5,
                          time_diffs=None,
                          use_time_proj=True):
        """Recursive implementation of curr_layers temporal graph attention layers.

        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert (n_layers >= 0)

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))

        source_node_features = self.node_features[source_nodes_torch, :]

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features

        if n_layers == 0:
            return source_node_features
        else:
            ############ 此处加上边的类别
            ### neighbours: [源节点, 评分等级, 邻居节点]
            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                                                                                          source_nodes,
                                                                                          timestamps,
                                                                                          n_neighbors=n_neighbors
                                                                                          )

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

            edge_deltas = timestamps[:, np.newaxis, np.newaxis] - edge_times

            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()
            ########## 在只有一层的情况下，返回邻居本身的embedding
            neighbor_embeddings = self.compute_embedding(memory,
                                                         neighbors,
                                                         np.repeat(timestamps, n_neighbors*n_relations),
                                                         n_layers=n_layers - 1,
                                                         n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), n_relations, effective_n_neighbors, -1)

            ##### 此处需要检查
            ## 张量展开
            edge_time_embeddings = self.time_encoder(edge_deltas_torch.view(len(source_nodes), -1))
            ## 展开后恢复
            edge_time_embeddings = edge_time_embeddings.view(len(source_nodes), n_relations,
                                                             effective_n_neighbors, -1)
            ## 没问题
            edge_features = self.edge_features[edge_idxs, :]

            mask = [neighbors_torch[:, r, :] == 0 for r in range(n_relations)]

            ################

            source_embedding = self.aggregate(n_layers, source_node_features,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

            return source_embedding

    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        return None


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                      neighbor_finder, time_encoder, n_layers,
                                                      n_node_features, n_edge_features,
                                                      n_time_features,
                                                      embedding_dimension, device,
                                                      n_heads, dropout,
                                                      use_memory)
        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            n_edge_features=n_edge_features,
            time_dim=n_time_features,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=n_node_features)
            for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding, _ = attention_model(source_node_features,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

        return source_embedding


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features,
                                       edge_features=edge_features,
                                       memory=memory,
                                       neighbor_finder=neighbor_finder,
                                       time_encoder=time_encoder,
                                       n_layers=n_layers,
                                       n_node_features=n_node_features,
                                       n_edge_features=n_edge_features,
                                       n_time_features=n_time_features,
                                       embedding_dimension=embedding_dimension,
                                       device=device,
                                       n_heads=n_heads, dropout=dropout, use_memory=use_memory)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))

