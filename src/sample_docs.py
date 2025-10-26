"""
Subset selection strategy
"""
import json
import logging
import sys
import os
import random
import numpy as np
import argparse
from data.dataset import WikitextDataset, BillDataset
from src.config import local_config
from src.utils import set_logging


class Selector:
    ''' At the moment we use only hard assignment, i.e. the assigned topic is the one with the highest weight. And for each cluster, we only consider those documents are hard assigned to it.'''
    def __init__(self, documents, doc2topic_weights):
        self.documents = documents
        self.indices = [i for i in range(len(documents))]
        self.doc2topic_weights = np.array(doc2topic_weights)
        row_sums = np.sum(self.doc2topic_weights, axis=1)
        self.non_outlier_mask = row_sums > 0
        self.non_outlier_indices = np.where(self.non_outlier_mask)[0]
        # Only normalize non-outlier rows to avoid division by zero
        self.doc2topic_weights[self.non_outlier_mask] = (
            self.doc2topic_weights[self.non_outlier_mask] /
            row_sums[self.non_outlier_mask].reshape(-1, 1)
        )
        # Set outlier rows to zero (optional, for clarity)
        self.doc2topic_weights[~self.non_outlier_mask] = 0

    def select(self, pos_k: int, neg_k, strategy):
        func_map: dict[str, callable] = {'random': self.random, 'top_k': self.top_k, 'weighted': self.weighted_sampling}
        selected = {}
        for topic_id in range(self.doc2topic_weights.shape[1]):
            pos_ids, _ = func_map[strategy](k=pos_k, t_id=topic_id)
            neg_ids, _ = self.negative(k=neg_k, t_id=topic_id)
            selected[topic_id] = {'pos': pos_ids.tolist(), 'neg': neg_ids.tolist()}
        return selected

    def random(self, k: int, t_id: int):
        topic_labels = np.argmax(self.doc2topic_weights, axis=1)
        topic_doc_ids = np.where((topic_labels == t_id) & self.non_outlier_mask)[0]
        k_final = min(k, len(topic_doc_ids))
        logging.info(f"random (hard): t_id={t_id}, k={k}, k_final={k_final}, num_valid={len(topic_doc_ids)}")
        if k_final == 0:
            return np.array([], dtype=int), []
        selected_ids = np.random.choice(topic_doc_ids, size=k_final, replace=False)
        selected_docs = [self.documents[i] for i in selected_ids]
        return selected_ids, selected_docs

    def top_k(self, k: int, t_id: int):
        topic_labels = np.argmax(self.doc2topic_weights, axis=1)
        topic_doc_ids = np.where((topic_labels == t_id) & self.non_outlier_mask)[0]
        weights = self.doc2topic_weights[topic_doc_ids, t_id]
        k_final = min(k, len(topic_doc_ids))
        logging.info(f"top_k (hard): t_id={t_id}, k={k}, k_final={k_final}, num_valid={len(topic_doc_ids)}")
        if k_final == 0:
            return np.array([], dtype=int), []
        sorted_indices = np.argsort(weights)
        selected_ids = topic_doc_ids[sorted_indices[-k_final:]]
        selected_docs = [self.documents[i] for i in selected_ids]
        return selected_ids, selected_docs
    
    def weighted_sampling(self, k: int, t_id: int):
        #selected_ids = np.random.choice(self.indices, size=k, p=self.doc2topic_weights[t_id], replace=False)
        logging.info(
        f"weighted_sampling: len(self.indices)={len(self.indices)}, "
        f"self.doc2topic_weights.shape={self.doc2topic_weights.shape}, "
        f"len(p)={len(self.doc2topic_weights[:, t_id])}, k={k}, t_id={t_id}"
    )
        topic_labels = np.argmax(self.doc2topic_weights, axis=1)
        topic_doc_ids = np.where((topic_labels == t_id) & self.non_outlier_mask)[0]
        weights = self.doc2topic_weights[topic_doc_ids, t_id]
        if len(topic_doc_ids) == 0 or weights.sum() == 0:
            return np.array([], dtype=int), []
        weights = weights / weights.sum()
        k_final = min(k, len(topic_doc_ids))
        logging.info(f"weighted_sampling (hard): t_id={t_id}, k={k}, k_final={k_final}, num_valid={len(topic_doc_ids)}")
        selected_ids = np.random.choice(topic_doc_ids, size=k_final, p=weights, replace=False)
        selected_docs = [self.documents[i] for i in selected_ids]
        return selected_ids, selected_docs

    def stratified(self, k, t_id):
        pass

    def utility(self, k, t_id):
        pass

    def negative(self, k, t_id):
        candidate_ids = set()
        num_topics = self.doc2topic_weights.shape[1]
        for other_topic in range(num_topics):
            if other_topic == t_id:
                continue
            # Top-k docs for this other topic
            sorted_indices = np.argsort(self.doc2topic_weights[:, other_topic])
            topk_indices = sorted_indices[-k:]
            candidate_ids.update(topk_indices)
        # Remove any docs that are also top-k for t_id (optional, for purity)
        topk_t_id = set(np.argsort(self.doc2topic_weights[:, t_id])[-k:])
        candidate_ids = list(candidate_ids - topk_t_id)
        if len(candidate_ids) == 0:
            return np.array([], dtype=int), []
        k_final = min(k, len(candidate_ids))
        selected_ids = random.sample(candidate_ids, k_final)
        selected_docs = [self.documents[i] for i in selected_ids]
        return np.array(selected_ids), selected_docs


if __name__ == '__main__':
    set_logging(log_file=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--pos_k', type=int)
    parser.add_argument('--neg_k', type=int)
    parser.add_argument('--topic_model', type=str)
    parser.add_argument('--num_topics', type=int)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    dataset = WikitextDataset(local_config['data'][args.dataset]) if args.dataset == 'wikitext' else BillDataset(local_config['data'][args.dataset])
    logging.info('Load dataset from {}'.format(local_config['data'][args.dataset]))
    saved_topic_model = f'output/topic_models/{args.topic_model}_{args.dataset}_{args.num_topics}.json'
    logging.info('Load topic model from {}'.format(saved_topic_model))
    logging.info('Number of Topics: {}, Sampling strategy: {}, pos_k: {}, neg_k: {}'.format(args.num_topics, args.strategy, args.pos_k, args.neg_k))
    with open(saved_topic_model, 'r') as fp:
        saved_weights = json.load(fp)

    selector = Selector(documents=dataset, doc2topic_weights=saved_weights['doc2topic_dist'])
    output = selector.select(pos_k=int(args.pos_k), neg_k=int(args.neg_k), strategy=args.strategy)

    with open(args.output_path, 'w') as fp:
        json.dump(output, fp)

