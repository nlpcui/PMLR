import sys
import os
import json
import argparse
import numpy as np
from data.dataset import WikitextDataset
from sentence_transformers import SentenceTransformer
from config import local_config
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


MODEL_ID = 'all-MiniLM-L6-v2'


class Scorer:
    def __init__(self, data, topic2doc_weights, doc2topic_weights, alpha, beta, num_topics):
        self.dataset = data
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        doc_texts = [data[i]['text'] for i in range(len(data))]
        self.embeddings = self.encoder.encode(doc_texts, show_progress_bar=True)
        self.sim_matrix = cosine_similarity(self.embeddings)

        self.topic2doc_weights = topic2doc_weights
        self.doc2topic_weights = doc2topic_weights
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics

    def score(self, output_path):
        # coverage
        topic_doc_coverage = []
        for topic_id in range(self.num_topics):
            weights = np.array(self.topic2doc_weights[topic_id])  # [n_doc]
            coverage = np.sum(weights * self.sim_matrix, axis=1)  # [n_doc]
            topic_doc_coverage.append(coverage)

        topic2sel = {}

        for topic_id in tqdm(range(self.num_topics)):
            selected_doc_ids = []
            in_topic_coverage_score = topic_doc_coverage[topic_id]  # [n_doc]
            cross_topic_coverage_score = np.mean([topic_doc_coverage[i] for i in range(self.num_topics) if i != topic_id], axis=0)  # [n_doc]
            redundancy_score = np.array([0. for i in range(len(self.dataset))])
            print('here', len(in_topic_coverage_score), len(cross_topic_coverage_score), len(redundancy_score))
            overall_score = in_topic_coverage_score - self.alpha * redundancy_score - self.beta * cross_topic_coverage_score

            while len(selected_doc_ids) < len(self.dataset):
                sel_id = int(np.argmax(overall_score))
                selected_doc_ids.append(sel_id)
                for i in range(len(redundancy_score)):
                    if self.sim_matrix[i, sel_id] > redundancy_score[i]:
                        redundancy_score[i] = self.sim_matrix[i, sel_id]
                # in_topic_coverage_score[sel_id] = float('-inf')
                overall_score = in_topic_coverage_score - self.alpha * redundancy_score - self.beta * cross_topic_coverage_score

            sorted_weights = sorted(self.topic2doc_weights[topic_id], reverse=True)
            for i in range(10):
                print(selected_doc_ids[i], self.topic2doc_weights[topic_id][selected_doc_ids[i]], sorted_weights[i])
            exit(1)
            topic2sel[topic_id] = selected_doc_ids

        with open(output_path, 'w') as fp_out:
            json.dump(topic2sel, fp_out)

        return topic2sel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--num_topics', type=int, default=25)
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--topic_model', type=str, default='lda')

    args = parser.parse_args()

    saved_weights = os.path.join('../output', 'topic_models', f'{args.topic_model}_{args.dataset}_{args.num_topics}.json')
    with open(saved_weights, 'r') as fp:
        saved_weights = json.load(fp)

    dataset = WikitextDataset(data_file=local_config['data']['wikitext'])
    scorer = Scorer(
        data=dataset,
        topic2doc_weights=saved_weights['topic2doc_dist'],
        doc2topic_weights=saved_weights['doc2topic_dist'],
        alpha=args.alpha,
        beta=args.beta,
        num_topics=args.num_topics
    )
    scorer.score(output_path=os.path.join('../output', 'score', f'{args.topic_model}_f{args.dataset}_T{args.num_topics}.json'))







