import math
import numpy as np
import torch


###################################################################################################
# 电影评分（未预训练）
def eval_score_prediction_beta(model, data, edge_idxs, batch_size, possible_score, n_neighbors):
    ground_truth = data.labels - 1
    pred_expectation = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        model.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx: e_idx]
            destinations_batch = data.destinations[s_idx: e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx: e_idx]

            pred_prob_batch = model(sources_batch, destinations_batch, destinations_batch,
                                    timestamps_batch, edge_idxs_batch, n_neighbors)
            pred_prob_batch = (torch.softmax(pred_prob_batch, dim=1) *
                               possible_score.view(1, -1)).sum(dim=1)
            pred_expectation[s_idx: e_idx] = pred_prob_batch.cpu().numpy()
    rmse = ((pred_expectation - ground_truth) ** 2.).mean().item()
    rmse = np.sqrt(rmse)
    return rmse
################################################################################################
