import sys
import os
import json
import argparse
import numpy as np
import logging
from data.dataset import Features
from pathlib import Path
from tqdm import tqdm


class DPPRetriever:
    def __init__(self, embeddings, relevance_scores):
        self.embeddings = embeddings
        self.relevance_scores = relevance_scores
        self.num_data, self.d = embeddings.shape

    def greedy_select(self, k):

        pass

    def kernel(self):
        pass


def greedy_dpp(embeddings: np.ndarray, relevance_scores: np.ndarray, alpha: float, num_select: int):
    """
    :param embeddings: [N, d]
    :param relevance_scores: [N, 1]
    :param alpha: factor
    :param num_select: number of selection
    :return:
    """

    num_items, dim = embeddings.shape
    weighted_embeddings = embeddings * relevance_scores[:, None]

    selected_indices = []
    remaining_indices = list(range(num_items))

    first_idx = np.argmax(relevance_scores)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    first_norm = np.sqrt(relevance_scores[first_idx])

    # Cholesky factor
    c_factor = np.array([[first_norm]])

    for t in tqdm(range(num_select-1)):
        deltas = []
        projection = []

        for idx in remaining_indices:
            # phi_i
            psi_i = weighted_embeddings[idx]

            # L_ii = phi_i * phi_i
            self_similarity = np.dot(psi_i, psi_i)

            # v = L_{S, i} = phi_S * phi_i
            psi_selected = weighted_embeddings[selected_indices]
            cross_similarity = psi_selected @ psi_i

            y = np.linalg.solve(c_factor.T, cross_similarity)

            delta_i = self_similarity - np.dot(y, y)
            deltas.append(delta_i)
            projection.append(y)

        best_pos = int(np.argmax(deltas))
        best_idx = remaining_indices[best_pos]
        best_y = projection[best_pos]
        best_delta = deltas[best_pos]

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        # update Cholesky
        size = c_factor.shape[0]
        new_row = np.zeros((1, size))
        c_factor = np.block([
            [c_factor, best_y.reshape(-1, 1)],
            [new_row, np.array([[np.sqrt(best_delta)]])]
        ])

    return selected_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,)
    parser.add_argument('--saved_features', type=str, )
    parser.add_argument('--saved_embeddings', type=str,)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--k', type=int,)
    parser.add_argument('--output_file', type=Path,)
    parser.add_argument('--feature_type', type=str)

    args = parser.parse_args()

    logging.info('Load features from {}'.format(args.saved_features))
    features = Features.load(feature_file=args.saved_features, feature_type=args.feature_type)

    with open(args.saved_embeddings, 'r') as fp_emb:
        embeddings = json.load(fp_emb)

    selected = {}
    for f_id, data_ids in features.label2data.items():
        feature_size = len(features.label2data[f_id])
        # print(np.max(features.feature2data_weights[f_id]), np.min(features.feature2data_weights[f_id]))
        # continue
        f_embs = np.array([embeddings[str(idx)] for idx in data_ids])
        f_rel = np.array([features.feature2data_weights[f_id][idx] for idx in data_ids])
        f_rel = np.exp(f_rel / (2 * args.alpha))
        # print('here', f_embs.shape, f_rel.shape)
        # exit(1)
        f_selected_pos = greedy_dpp(embeddings=f_embs, relevance_scores=f_rel, alpha=args.alpha, num_select=min(args.k, feature_size))
        selected[f_id] = {'pos': {int(data_ids[pos]): float(f_rel[pos]) for pos in f_selected_pos}, 'neg': []}

    with open(args.output_file, 'w') as fp_out:
        json.dump(selected, fp_out)

