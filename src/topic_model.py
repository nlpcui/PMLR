"""
Topic/Clustering model
"""
import json
import logging
import argparse
import sys
import os

import nltk
import tomotopy as tp
import numpy as np
from data.dataset import WikitextDataset, BillDataset
from src.utils import set_logging
from src.config import local_config
from collections import OrderedDict
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


def train_lda(corpus, num_topics, save_path, max_iterations=1000, step=10):
    logging.info('Train LDA model')
    model = tp.LDAModel(k=num_topics)
    for data in corpus:
        model.add_doc(data['tokenized_text'])

    logging.info('{} docs, {} topics'.format(len(corpus), num_topics))
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


class TopicModelEvaluator:
    def __init__(self, dataset, output):
        self.dataset = dataset
        self.output = output

    def purity(self):
        purity = {}
        all_true_labels = []
        for data in self.dataset:
            pred_topic_id = int(np.argmax(self.output['doc2topic_dist'][data['id']]))
            true_topic_label = data['label']
            if true_topic_label not in all_true_labels:
                all_true_labels.append(true_topic_label)
            # purity (pred, true)
            if pred_topic_id not in purity:
                purity[pred_topic_id] = {}
            if true_topic_label not in purity[pred_topic_id]:
                purity[pred_topic_id][true_topic_label] = 0
            purity[pred_topic_id][true_topic_label] += 1

        purity_scores = []
        for pred_id in purity:
            # print('here', purity_pred_true[pred_id])
            purity_scores.append(max(purity[pred_id].values()) / sum(purity[pred_id].values()))
        # print('here', purity_scores)
        purity_score = sum(purity_scores) / len(purity_scores)

        inverse_purity_score = 0

        for true_label in all_true_labels:
            inverse_purity_score += max([v.get(true_label, 0) for v in purity.values()])
        inverse_purity_score /= len(self.dataset)

        return {'inverse_purity': inverse_purity_score, 'purity': purity_score, 'harmonic': 2 / (1 / purity_score + 1 / inverse_purity_score)}

    def c_npmi(self):
        # print('here')
        # print(len(tokenized_doc))
        # print(tokenized_doc[0])
        # print('t', topic_words)
        # exit(1)
        tokenized_doc = [text.lower().split(' ') for text in self.dataset.texts]
        topic_words = [[self.output['vocab'][word_id] for word_id in np.argsort(weight)[-10:]] for weight in self.output['topic2word_dist']]
        dictionary = Dictionary(tokenized_doc)

        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized_doc,
            dictionary=dictionary,
            coherence='c_npmi'
        )

        npmi = coherence_model.get_coherence()
        return npmi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)   # [train, eval]
    parser.add_argument('--dataset', type=str, default='topic_wikitext')
    parser.add_argument('--num_topics', type=int, default=25)
    parser.add_argument('--model_type', type=str, default='lda')
    parser.add_argument('--max_iterations', type=int)

    args = parser.parse_args()
    set_logging(log_file=None)

    if args.dataset == 'topic_wikitext':
        documents = WikitextDataset(data_file=local_config['data'][args.dataset])
    else:
        documents = BillDataset(data_file=local_config['data'][args.dataset])

    if args.task == 'train':
        if args.model_type == 'lda':
            train_lda(corpus=documents, num_topics=args.num_topics, max_iterations=args.max_iterations, save_path=f'output/topic_models/{args.model_type}_{args.dataset}_{args.num_topics}.json')
        else:
            pass

    elif args.task == 'eval':
        with open(f'output/topic_models/{args.model_type}_{args.dataset}_{args.num_topics}.json', 'r') as fp_output:
            topic_model_output = json.load(fp_output)
        evaluator = TopicModelEvaluator(dataset=documents, output=topic_model_output)
        result = evaluator.c_npmi()
        print('Dataset: {}, num_topics: {}, NPMI: {}'.format(args.dataset, args.num_topics, result))
