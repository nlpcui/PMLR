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
from data.dataset import WikitextDataset, BillDataset, SAEWikitext, Features
from src.config import local_config
from src.utils import set_logging
from pathlib import Path


class Sampler:
    def __init__(self, data, features):
        self.data = data
        self.features = features
        self.indices = [i for i in range(len(data))]  # [len(data)]
        # [num_features/topics, ]

    def sample(self, method, params):
        if method == 'random':
            pos_samples = self.random(k=params['pos_k'])
        elif method == 'stratified':
            pos_samples = self.stratified(k=params['pos_k'], num_strata=params['num_strata'])
        elif method == 'topk':
            pos_samples = self.top_k(k=params['pos_k'])
        elif method == 'weighted':
            pos_samples = self.weighted(k=params['pos_k'])

        # always random
        neg_samples = self.sample_negative(k=params['neg_k'])

        return {f_id: {'pos': pos_samples[f_id], 'neg': neg_samples[f_id]} for f_id in range(self.features.num_features)}

    def random(self, k: int):  # todo:
        sampled = {}
        for f_id in range(self.features.num_features):
            data_ids = self.features.label2data[f_id]
            # if len(data_ids) < self.min_feature_size:
            #     continue
            sampled_ids = np.random.choice(data_ids, size=min(k, len(data_ids)), replace=False)
            # sampled_items = [self.data[i] for i in sampled_ids]
            sampled_weights = self.features.feature2data_weights[f_id][sampled_ids]
            sampled[f_id] = dict(zip(sampled_ids.tolist(), sampled_weights))
        return sampled

    def top_k(self, k: int):
        sampled = {}
        for f_id in range(self.features.num_features):
            # if len(self.features.label2data[f_id]) < self.min_feature_size:
            #     continue
            sorted_ids = np.argsort(self.features.feature2data_weights[f_id])
            sampled_ids = sorted_ids[-k:]
            # sampled_data = [self.data[i] for i in sampled_ids]
            sampled_weights = self.features.feature2data_weights[f_id][sampled_ids]
            sampled[f_id] = dict(zip(sampled_ids.tolist(), sampled_weights))
        return sampled

    def weighted(self, k: int):
        sampled = {}
        for f_id in range(self.features.num_features):
            f_data_id = self.features.label2data[f_id]
            # if len(f_data_id) < self.min_feature_size:
            #     continue
            f_data_weights = self.features.feature2data_weights[f_id][f_data_id]
            sampled_ids = np.random.choice(f_data_id, size=min(k, len(f_data_id)), p=f_data_weights / np.sum(f_data_weights), replace=False)
            # sampled_data = [self.data[i] for i in sampled_ids]
            sampled_weights = self.features.feature2data_weights[f_id][sampled_ids]
            sampled[f_id] = dict(zip(sampled_ids.tolist(), sampled_weights))
        return sampled

    def stratified(self, k: int, num_strata: int):
        sampled = {}
        for f_id in range(self.features.num_features):
            f_data_id = self.features.label2data[f_id]
            # if len(f_data_id) < self.min_feature_size:
            #     continue
            f_data_weights = self.features.feature2data_weights[f_id][f_data_id]
            sorted_f_data_id = f_data_id[np.argsort(f_data_weights)]
            chunks = np.array_split(sorted_f_data_id, min(num_strata, k))
            sampled_ids = []
            for chunk in chunks:
                chunk_samples = np.random.choice(chunk, size=min(k//min(num_strata, k), len(chunk)), replace=False)
                sampled_ids.extend(chunk_samples.tolist())
            sampled_weights = self.features.feature2data_weights[f_id][sampled_ids]
            sampled[f_id] = dict(zip(sampled_ids, sampled_weights))
        return sampled

    def sample_negative(self, k):
        sampled = {}
        for f_id in range(self.features.num_features):
            data_ids = []
            for data_id in self.features.data2labels:
                if f_id not in self.features.data2labels[data_id]:
                    data_ids.append(data_id)
            sampled[f_id] = np.random.choice(data_ids, size=min(k, len(data_ids)), replace=False).tolist()
        return sampled

    # def negative(self, k, t_id):
    #     candidate_ids = set()
    #     for topic_id, weights in enumerate(self.doc2topic_weights):
    #         if topic_id == t_id:
    #             continue
    #         sorted_ids = np.argsort(weights)
    #         candidate_ids.update(sorted_ids[-k:])
    #
    #     candidate_ids = list(candidate_ids)
    #     # print('here', len(self.doc2topic_weights[t_id]), sum(self.doc2topic_weights[t_id]))
    #     # exit(1)
    #     selected_ids = np.random.choice(candidate_ids, size=k, p=self.doc2topic_weights[t_id], replace=False)
    #     selected_docs = [self.documents[i] for i in selected_ids]
    #     return selected_ids, selected_docs
    #
    # def negative_old(self, k, t_id):
    #     sorted_indices = np.argsort(self.doc2topic_weights[t_id])
    #     selected_ids = sorted_indices[: k]
    #     selected_docs = [self.documents[i] for i in range(k)]
    #     return selected_ids, selected_docs


if __name__ == '__main__':
    set_logging(log_file=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wiki')
    parser.add_argument('--sample_method', type=str)
    parser.add_argument('--pos_k', type=int)
    parser.add_argument('--neg_k', type=int)
    parser.add_argument('--task', type=str)
    parser.add_argument('--num_features', type=int, default=25)
    parser.add_argument('--output_path', type=str, )
    parser.add_argument('--num_strata', type=int, default=10)
    parser.add_argument('--saved_features', type=Path)
    parser.add_argument('--feature_type', type=str)

    args = parser.parse_args()

    logging.info('Load dataset from {}'.format(local_config['data'][args.dataset]))
    if args.dataset == 'topic_wikitext':
        dataset = WikitextDataset(local_config['data'][args.dataset])
    elif args.dataset == 'topic_bill':
        dataset = BillDataset(local_config['data'][args.dataset])
    elif args.dataset == 'sae_wikitext':
        dataset = SAEWikitext(local_config['data'][args.dataset])

    # features
    logging.info('Load features from {}'.format(args.saved_features))
    # saved_features = f'output/topic_models/{args.topic_model}_{args.dataset}_{args.num_features}.json'
    features = Features.load(feature_file=args.saved_features, feature_type=args.feature_type)

    sampler = Sampler(data=dataset, features=features)
    output = sampler.sample(method=args.sample_method, params={'pos_k': int(args.pos_k), 'neg_k': int(args.neg_k), 'num_strata': args.num_strata})
    print(args.dataset, args.sample_method, args.pos_k, output)
    with open(args.output_path, 'w') as fp_out:
        json.dump(output, fp_out)

