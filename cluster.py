"""
Topic/Clustering model
"""
import json
import logging
import argparse
import sys
import os
import tomotopy as tp
import numpy as np
from data.dataset import WikitextDataset, BillDataset
from utils import set_logging
from collections import OrderedDict


def train_lda(corpus, num_topics, save_path, max_iterations=1000, step=10):
    logging.info('Train LDA model')
    model = tp.LDAModel(k=num_topics)
    for data in corpus:
        model.add_doc(data['tokenized_text'])

    all_ll = []
    for i in range(0, max_iterations, step):
        model.train(step)
        all_ll.append(model.ll_per_word)
        logging.info('Iteration: {}\tLog-likelihood: {}'.format(i, model.ll_per_word))
        if len(all_ll) > 3 and all_ll[-1] < all_ll[-2] < all_ll[-3]:
            break

    vocab = list(model.used_vocabs)

    model.summary()

    topic2word = np.array([model.get_topic_word_dist(topic_id) for topic_id in range(model.k)])   # [T, V]

    # doc-topic dist
    doc2topic = np.array([doc.get_topic_dist() for doc in model.docs])    # [N, T]
    # topic-doc dist
    topic_sum = doc2topic.sum(axis=0, keepdims=True)  # T
    topic2doc = doc2topic / topic_sum  # [N, T]
    topic2doc = topic2doc.T

    with open(save_path, 'w') as fp:
        json.dump({'topic2word_dist': topic2word.tolist(), 'doc2topic_dist': doc2topic.tolist(), 'topic2doc_dist': topic2doc.tolist(), 'vocab': vocab}, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_topics', type=int)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--max_iterations', type=int)

    args = parser.parse_args()
    set_logging(log_file=None)

    if args.dataset == 'wikitext':
        documents = WikitextDataset(data_file='data/wikitext/train.metadata.jsonl')
    else:
        documents = BillDataset(data_file='data/bill/train.metadata.jsonl')

    train_lda(corpus=documents, num_topics=args.num_topics, max_iterations=args.max_iterations, save_path=f'output/topic_models/{args.model_type}_{args.dataset}_{args.num_topics}.json')
