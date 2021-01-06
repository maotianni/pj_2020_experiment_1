import torch
from torch import nn

from utils.utils import MergeLayer


class TemporalAttentionLayer(torch.nn.Module):
    def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim,
                 output_dimension, n_head=2, n_relations=5,
                 dropout=0.1, aggregate='stack'):
        super(TemporalAttentionLayer, self).__init__()

        self.n_head = n_head

        self.n_relations = n_relations

        self.feat_dim = n_node_features
        self.time_dim = time_dim

        self.query_dim = n_node_features + time_dim
        self.key_dim = n_neighbors_features + time_dim + n_edge_features

        self.aggregate = aggregate
        if self.aggregate == 'stack':
            self.merger = MergeLayer(self.query_dim * n_relations, n_node_features,
                                     n_node_features, output_dimension)
        elif self.aggregate == 'sum':
            self.merger = MergeLayer(self.query_dim, n_node_features,
                                     n_node_features, output_dimension)
        else:
            raise ValueError("Aggregating Method {} not supported".format(self.aggregate))

        self.multi_head_target = torch.nn.ModuleList([nn.MultiheadAttention(embed_dim=self.query_dim,
                                                                            kdim=self.key_dim,
                                                                            vdim=self.key_dim,
                                                                            num_heads=n_head,
                                                                            dropout=dropout)
                                                                            for _ in range(n_relations)])

    def forward(self, src_node_features, src_time_features, neighbors_features,
                neighbors_time_features, edge_features, neighbors_padding_mask):
        """
        "Temporal attention model
        :param src_node_features: float Tensor of shape [batch_size, n_node_features]
        :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
        :param neighbors_features: float Tensor of shape [batch_size, n_relations, n_neighbors, n_node_features]
        :param neighbors_time_features: float Tensor of shape [batch_size, n_relations, n_neighbors, time_dim]
        :param edge_features: float Tensor of shape [batch_size, n_relations, n_neighbors, n_edge_features]
        :param neighbors_padding_mask: float Tensor of shape [batch_size, n_relations, n_neighbors]
        :return:
        attn_output: float Tensor of shape [1, batch_size, n_node_features]
        attn_output_weights: [batch_size, 1, n_neighbors]
        """

        src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)

        query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)
        query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]

        all_attn_out = []
        all_attn_weights = []
        for r in range(self.n_relations):
            key = torch.cat([neighbors_features[:,r,:,:],
                             edge_features[:,r,:,:],
                             neighbors_time_features[:,r,:,:]], dim=2)
            key = key.permute([1, 0, 2])

            # Compute mask of which source nodes have no valid neighbors
            invalid_neighborhood_mask = neighbors_padding_mask[r].all(dim=1, keepdim=True)
            # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
            # force the attention to just 'attend' on this neighbor (which has the same features as all
            # the others since they are fake neighbors) and will produce an equivalent result to the
            # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
            neighbors_padding_mask[r][invalid_neighborhood_mask.squeeze(), 0] = False

            attn_output, attn_output_weights = self.multi_head_target[r](query=query, key=key, value=key,
                                                                      key_padding_mask=neighbors_padding_mask[r])

            attn_output = attn_output.squeeze()
            attn_output_weights = attn_output_weights.squeeze()

            attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
            attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

            all_attn_out.append(attn_output)
            all_attn_weights.append(attn_output_weights)

        if self.aggregate == 'stack':
            all_attn_out = torch.cat(all_attn_out, dim=1)
        elif self.aggregate == 'sum':
            all_attn_out = sum(all_attn_out)
        else:
            raise ValueError("Aggregating Method {} not supported".format(self.aggregate))

        all_attn_out = self.merger(all_attn_out, src_node_features)

        return all_attn_out, all_attn_weights
