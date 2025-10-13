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
    def __init__(self, documents, doc2topic_weights):
        self.documents = documents
        self.indices = [i for i in range(len(documents))]
        self.doc2topic_weights = np.array(doc2topic_weights)

        self.doc2topic_weights = self.doc2topic_weights / np.sum(self.doc2topic_weights, axis=1).reshape(-1, 1)

    def select(self, pos_k: int, neg_k, strategy):
        func_map: dict[str, callable] = {'random': self.random, 'top_k': self.top_k, 'weighted': self.weighted_sampling}
        selected = {}
        for topic_id in range(len(self.doc2topic_weights)):
            pos_ids, _ = func_map[strategy](k=pos_k, t_id=topic_id)
            neg_ids, _ = self.negative(k=neg_k, t_id=topic_id)
            selected[topic_id] = {'pos': pos_ids.tolist(), 'neg': neg_ids.tolist()}
        return selected

    def random(self, k: int, t_id: int):  # todo:
        topic_labels = np.argmax(self.doc2topic_weights, axis=1)
        topic_doc_ids = np.where(topic_labels == t_id)[0]
        selected_ids = random.sample(topic_doc_ids, k)
        selected_docs = [self.documents[i] for i in selected_ids]
        return selected_ids, selected_docs

    def top_k(self, k: int, t_id: int):
        sorted_indices = np.argsort(self.doc2topic_weights[t_id])
        selected_ids = sorted_indices[-k:]
        selected_docs = [self.documents[i] for i in selected_ids]
        return selected_ids, selected_docs

    def weighted_sampling(self, k: int, t_id: int):
        selected_ids = np.random.choice(self.indices, size=k, p=self.doc2topic_weights[t_id], replace=False)
        selected_docs = [self.documents[i] for i in selected_ids]
        return selected_ids, selected_docs

    def stratified(self, k, t_id):
        pass

    def utility(self, k, t_id):
        pass

    def negative(self, k, t_id):
        candidate_ids = set()
        for topic_id, weights in enumerate(self.doc2topic_weights):
            if topic_id == t_id:
                continue
            sorted_ids = np.argsort(weights)
            candidate_ids.update(sorted_ids[-k:])

        candidate_ids = list(candidate_ids)
        # print('here', len(self.doc2topic_weights[t_id]), sum(self.doc2topic_weights[t_id]))
        # exit(1)
        selected_ids = np.random.choice(candidate_ids, size=k, p=self.doc2topic_weights[t_id], replace=False)
        selected_docs = [self.documents[i] for i in selected_ids]
        return selected_ids, selected_docs

    def negative_old(self, k, t_id):
        sorted_indices = np.argsort(self.doc2topic_weights[t_id])
        selected_ids = sorted_indices[: k]
        selected_docs = [self.documents[i] for i in range(k)]
        return selected_ids, selected_docs


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
    with open(saved_topic_model, 'r') as fp:
        saved_weights = json.load(fp)

    selector = Selector(documents=dataset, doc2topic_weights=saved_weights['topic2doc_dist'])
    output = selector.select(pos_k=int(args.pos_k), neg_k=int(args.neg_k), strategy=args.strategy)

    with open(f'output/sampled/{args.output_path}', 'w') as fp:
        json.dump(output, fp)

