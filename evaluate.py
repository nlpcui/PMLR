import json
import argparse
import logging

import numpy as np
from openai import OpenAI
from utils import set_logging
from config import local_config
from data.dataset import WikitextDataset, BillDataset
from sampling import Selector
from string import Template
from config import openai_key
from tqdm import tqdm

prompt_eval = Template(
"""
You are given a document and one or more topics. For each topic, determine whether the topic applies to the document based on the following rules:
1. The topic directly matches the subject or main focus of the document. Loosely or indirectly related topics should not be considered applicable (e.g., basketball is not applicable to a document about football).
2. The topic is a super topic — that is, a broader category of the document’s subject (e.g., “sports” is a super topic of “basketball”).

If the topic contains factual details that are not presented in the document, then mark is as not applicable. For example, if a topic is "NBA Teams in 1990s" while the document does not mention "1990s", it should be considered not applicable. 

Below is the input documents and topics:
Document: $document

Topics: 
$topics

Output format:
Provide your answer in the following JSON format:
{
"<topic_id>": <1 if applicable otherwise 0>,
"<topic_id>": <1 if applicable otherwise 0>,
...
}

Directly provide the JSON output without any other output including the format code like ```json```, etc. 

Output:""")


class TopicLabelEvaluator:
    def __init__(self, annotated_clusters, annotated_documents, dataset, topic2doc_weights, prompt_model_id, prompt_template, temperature, selector, sample_per_cluster, max_retry=3):
        """
        :param dataset:
        :param topic2doc_weights: [T, N] float matrix
        :param prompt_model_id:
        :param temperature: None or a K
        """

        assert len(annotated_clusters) == len(topic2doc_weights) == len(annotated_documents)

        self.prompt_template = prompt_template
        self.prompt_model_id = prompt_model_id
        self.temperature = temperature

        self.dataset = dataset
        self.topic2doc_weights = topic2doc_weights
        self.max_retry = max_retry
        self.selector = selector
        self.client = OpenAI(api_key=openai_key)
        self.annotated_clusters = annotated_clusters
        self.annotated_documents = annotated_documents
        self.sample_per_cluster = sample_per_cluster

        # eval_results: {doc_id: {topic_label: "label", topic_id: id, match: [] }}
        self.eval_result = {}
        self.score_matrix = np.zeros((len(topic2doc_weights), len(topic2doc_weights)))
        self.sampled_topic_weights = np.zeros((len(topic2doc_weights), ))
        self.coverage = []
        self.disc = []

    def evaluate(self, output_path):
        eval_result = {}
        for topic_id, weights in self.topic2doc_weights:
            # sample K documents per topic
            docs = self.selector.select(weights, k=self.sample_per_cluster)
            # eval doc - topic match
            for doc in docs:
                match = self.__eval_single_doc__(doc, labels)
                self.eval_result[doc['id']] = {'topic_id': topic_id, 'topic_label': self.annotated_clusters[topic_id], 'match': match}

        self.__calculate_score__()

    def __eval_single_doc__(self, doc, labels):
        prompt = self.prompt_template.substitute(document=doc, topics=labels)
        messages = [
            {'role': 'system', 'content': 'You are a help content analyst that helps me analyze the topics discussed in documents.'},
            {'role': 'user', 'content': prompt}
        ]

        # print('prompt', prompt)
        # exit(1)
        for i in range(self.max_retry):
            response = self.client.chat.completions.create(model=self.prompt_model_id, messages=messages, temperature=self.temperature)
            output = response.choices[0].message.content

            try:
                match = json.loads(output)
                return match
            except Exception:
                continue
        return

    def __calculate_score__(self):
        for doc_id, doc_eval in self.eval_result.items():
            self.score_matrix[doc_eval['topic_id']] += np.array(doc_eval['match']) * self.topic2doc_weights[:, doc_id]

        self.score_matrix /= self.sampled_topic_weights

        self.coverage = [self.score_matrix[i][i] for i in range(len(self.topic2doc_weights))]
        self.disc = [(sum(self.score_matrix[i]) - self.score_matrix[i][i]) / (len(self.topic2doc_weights) - 1) for i in range(self.topic2doc_weights)]

    def eval_consistency(self, output_path):
        eval_result = {}
        with tqdm(total=len(self.annotated_clusters) * len(self.annotated_documents[0])) as pbar:
            for topic_id, annotation in enumerate(self.annotated_clusters):
                cluster_label = annotation['topic_name']
                for eval_doc_id in self.annotated_documents[topic_id]:
                    # print('here ${}$'.format(int(eval_doc_id)), type(self.dataset))
                    try:
                        doc = self.dataset[int(eval_doc_id)]  # ValueError: invalid literal for int() with base 10: 'IX Corps ( United States )'
                        # print(eval_doc_id, doc['text'])
                        # exit(1)
                        if eval_doc_id not in eval_result:
                            eval_result[eval_doc_id] = {}
                        # cluster consistency
                        clu_res = self.__eval_single_doc__(doc=doc['text'], labels='Topic ID 0: {}'.format(cluster_label))
                        # print('cluster output', clu_res)
                        # exit(1)
                        eval_result[eval_doc_id][cluster_label] = {'match': clu_res['0'], 'topic_level': 'cluster'}
                        # individual consistency
                        # print('This', list(self.annotated_documents[0].keys()), type(list(self.annotated_documents[0].keys())[0]))
                        doc_labels = [item['topic_name'] for item in self.annotated_documents[topic_id][eval_doc_id]]
                        ind_res = self.__eval_single_doc__(doc=doc['text'], labels='\n'.join(['Topic ID {}: {}'.format(i, t) for i, t in enumerate(doc_labels)]))
                        # print('topic output', ind_res)
                        for doc_tid in range(len(doc_labels)):
                            eval_result[eval_doc_id][doc_labels[doc_tid]] = {'match': ind_res[str(doc_tid)], 'topic_level': 'document'}
                        # print('One round', eval_result)
                        # exit(1)
                    except Exception:
                        continue
                    pbar.update(1)

        # document-level consistency
        doc_match_cnt = 0
        cluster_match_cnt = 0

        for doc_id in eval_result:
            for topic in eval_result[doc_id]:
                if eval_result[doc_id][topic]['topic_level'] == 'document':
                    doc_match_cnt += eval_result[doc_id][topic]['match']
                else:
                    cluster_match_cnt += eval_result[doc_id][topic]['match']

        cluster_consistency = cluster_match_cnt / len(eval_result)
        doc_consistency = doc_match_cnt / sum([len(eval_result[doc_id]) - 1 for doc_id in eval_result])
        print('cluster consistency {}, doc consistency {}'.format(cluster_consistency, doc_consistency))
        with open(output_path, 'w') as fp_output:
            json.dump({'cluster_consistency': cluster_consistency, 'doc_consistency': doc_consistency, 'full_eval': eval_result}, fp_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)                      # [wikitext, bill]
    parser.add_argument('--saved_weights', type=str)
    parser.add_argument('--saved_annotations', type=str)
    parser.add_argument('--prompt_model_id', type=str)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--eval_sample_per_cluster', type=int)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--task', type=str)                         # [consistency, metric]

    args = parser.parse_args()

    # dataset
    assert args.dataset in ['wikitext', 'bill']

    if args.dataset == 'wikitext':
        corpus = WikitextDataset(data_file=local_config['data']['wikitext'])
    else:
        corpus = BillDataset(data_file=local_config['data']['bill'])

    # annotation result
    with open(args.saved_annotations, 'r') as fp:
        saved_annotations = json.load(fp)

    # saved weights
    with open(args.saved_weights, 'r') as fp:
        saved_weights = json.load(fp)

    # sample config
    evaluator = TopicLabelEvaluator(
        annotated_clusters=[saved_annotations[str(topic_id)]['common_topic'] for topic_id in range(len(saved_annotations))],
        annotated_documents=[saved_annotations[str(topic_id)]['document_topics'] for topic_id in range(len(saved_annotations))],
        dataset=corpus,
        topic2doc_weights=np.array(saved_weights['topic2doc_dist']),
        prompt_model_id=args.prompt_model_id,
        prompt_template=prompt_eval,
        temperature=args.temperature,
        selector=Selector(dataset=corpus),
        sample_per_cluster=args.eval_sample_per_cluster,
        max_retry=3
    )

    if args.task == 'consistency':
        evaluator.eval_consistency(output_path=args.output_path)
    elif args.task == 'metric':
        evaluator.evaluate(output_path=args.output_path)
    else:
        logging.error('Unknown task')
