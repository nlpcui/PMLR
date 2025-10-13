import json
import argparse
import os
import sys
import logging
import traceback
import numpy as np
import random
from openai import OpenAI
from src.config import local_config
from data.dataset import WikitextDataset, BillDataset
from src.sample_docs import Selector
from string import Template
from src.config import openai_key, api_price
from tqdm import tqdm
from dataclasses import dataclass
from src.utils import set_logging, truncate
from pathlib import Path

prompt_eval = Template(
"""
You are given a topic and some documents. For each document, determine whether the topic applies to it based on the following rules:
1. The topic directly matches the subject or main focus of the document. Loosely or indirectly related topics should not be considered applicable (e.g., basketball is not applicable to a document about football).
2. The topic is a super topic — that is, a broader category of the document’s subject (e.g., “sports” is a super topic of “basketball”).

If the topic contains factual details that are not presented in the document, then mark is as not applicable. For example, if a topic is "NBA Teams in 1990s" while the document does not mention "1990s", it should be considered not applicable. 

Below is the input documents and topics:
$topic

Documents: 
$documents

Output format:
Provide your answer in the following JSON format:
{
"<document_id>": <1 if applicable otherwise 0>,
"<document_id>": <1 if applicable otherwise 0>,
...
}

Directly provide the JSON output without any other output including the format code like ```json```, etc. 
Note that you must include all <document_id> in your output, even for documents that do not match to the topic (set their value to 0). 

Output:""")


@dataclass
class Cluster:
    doc_weights: list
    label: str
    documents: set
    sampled_docs_ann: list
    sampled_docs_eval: list


@dataclass
class Document:
    topic_weights: list
    label_id: int
    topic_ann: list
    match_ann: list


def is_int(x):
    try:
        x = int(x)
        return True
    except Exception:
        return False

#
# class TopicLabelEvaluator:
#     def __init__(self, num_topics, annotations, weights, dataset, prompt_model_id, prompt_template, temperature, selector, eval_sample_per_cluster, ann_sample_per_cluster, max_retry=3):
#         """
#         :num_topics
#         :param dataset:
#         :param prompt_model_id:
#         :param temperature: None or a K
#         """
#
#         assert num_topics == len(annotations), f"annotations: {len(annotations)}, num_topics: {num_topics}"
#
#         self.num_topics = num_topics
#
#         self.topics = {}
#         self.documents = {}
#
#         for doc_id in range(len(weights['doc2topic_dist'])):
#             doc_id = int(doc_id)
#             self.documents[int(doc_id)] = Document(
#                 topic_weights=weights['doc2topic_dist'][doc_id],
#                 label_id=int(np.argmax(weights['doc2topic_dist'][doc_id])),
#                 topic_ann=[],
#                 match_ann=[],
#             )
#
#         for topic_id in range(num_topics):
#
#             self.topics[topic_id] = Cluster(
#                     label=annotations[str(topic_id)]['common_topic']['topic_name'],
#                     doc_weights=weights['topic2doc_dist'][topic_id],
#                     sampled_docs_ann=[int(doc_id) for doc_id in annotations[str(topic_id)]['document_topics'] if is_int(doc_id)],
#                     sampled_docs_eval=[],
#                     documents=set()
#             )
#             for doc_id in self.topics[topic_id].sampled_docs_ann:
#                 self.documents[int(doc_id)].topic_ann.extend(annotations[str(topic_id)]['document_topics'][str(doc_id)])
#
#         for doc_id in self.documents:
#             self.topics[self.documents[doc_id].label_id].documents.add(doc_id)
#
#         self.eval_sample_per_cluster = eval_sample_per_cluster
#         self.ann_sample_per_cluster = ann_sample_per_cluster
#         self.prompt_template = prompt_template
#         self.prompt_model_id = prompt_model_id
#         self.temperature = temperature
#
#         self.dataset = dataset
#         self.max_retry = max_retry
#         self.selector = selector
#         self.client = OpenAI(api_key=openai_key)
#
#         # eval_results: {doc_id: {topic_label: "label", topic_id: id, match: [] }}
#         self.eval_result = {}
#         # self.score_matrix = np.zeros((len(topic2doc_weights), len(topic2doc_weights)))
#         # self.sampled_topic_weights = np.zeros((len(topic2doc_weights), ))
#         self.coverage = []
#         self.disc = []
#
#     def eval_metric(self, output_path):
#         eval_result = {topic_id: {'doc_match': None, 'in_topic_match': None, 'cross_topic_match': None} for topic_id in self.topics}
#
#         labels = [self.topics[i].label for i in self.topics]
#         # evaluate match
#         total_cost = 0
#         fail_cnt = 0
#         with tqdm(total=len(self.topics) * self.eval_sample_per_cluster) as pbar:
#             for topic_id in self.topics:
#                 # sample K documents per topic
#                 doc_weights = [self.topics[topic_id].doc_weights[doc_id] if doc_id in self.topics[topic_id].documents else 0 for doc_id in self.documents]
#                 doc_weights = [v / sum(doc_weights) for v in doc_weights]
#                 selected_doc_ids = self.selector.select(doc_weights=doc_weights, k=self.eval_sample_per_cluster)
#                 # print('here', selected_doc_ids)
#                 self.topics[topic_id].sampled_docs_eval = selected_doc_ids
#                 # eval inter doc-topic match
#                 for doc_id in selected_doc_ids:
#                     attempt = 0
#                     match_res = []
#                     while attempt < self.max_retry:
#                         match, cost = self.__eval_single_doc__(doc_text=self.dataset[doc_id]['text'], labels='\n'.join(['Topic ID {}: {}'.format(i, labels[i]) for i in range(len(labels))]))
#                         total_cost += cost
#                         match = list(match.values())
#                         if len(match) == self.num_topics and all([is_int(item) for item in match]) and all([int(item) in [0, 1] for item in match]):
#                             match = [int(item) for item in match]
#                             match_res = match
#                             break
#                         else:
#                             fail_cnt += 1
#                     pbar.set_description('Cost {}, fail cnt {}'.format(total_cost, fail_cnt))  # 动态设置描述
#                     pbar.update(1)
#                     self.documents[doc_id].match_ann = match_res
#                     eval_result[topic_id]['doc_match'] = match_res
#
#         # calculate_score
#         for topic_id in self.topics:
#             # in topic match
#             in_topic_match = 0
#             weight_sum = 0
#             for doc_id in self.topics[topic_id].sampled_docs_eval:
#                 # print('that', doc_id, topic_id, len(self.topics[topic_id].doc_weights), len(self.documents), )
#                 # print('this', len(self.documents[doc_id].match_ann), self.documents[doc_id].match_ann)
#                 if not self.documents[doc_id].match_ann:    # validity check
#                     continue
#                 in_topic_match += self.topics[topic_id].doc_weights[doc_id] * self.documents[doc_id].match_ann[topic_id]
#                 weight_sum += self.topics[topic_id].doc_weights[doc_id]
#             in_topic_match /= weight_sum
#             eval_result[topic_id]['in_topic_match'] = in_topic_match
#
#             # cross topic match
#             cross_topic_match = {}
#             for t in self.topics:
#                 if t == topic_id:
#                     continue
#                 cross_topic_match[t] = 0
#                 weight_sum = 0
#                 for doc_id in self.topics[topic_id].sampled_docs_eval:
#                     if not self.documents[doc_id].match_ann:
#                         continue    # validity check
#                     cross_topic_match[t] += self.topics[t].doc_weights[doc_id] * self.documents[doc_id].match_ann[t]
#                     weight_sum += self.topics[t].doc_weights[doc_id]
#                 cross_topic_match[t] /= weight_sum
#
#             eval_result[topic_id]['cross_topic_match'] = cross_topic_match
#
#         with open(output_path, 'w') as fp_out:
#             json.dump(eval_result, fp_out)
#
#         print('here', eval_result)
#         exit(1)
#
#     def __eval_single_doc__(self, doc_text, labels):
#         prompt = self.prompt_template.substitute(document=doc_text, topics=labels)
#         messages = [
#             {'role': 'system', 'content': 'You are a content analyst that helps me analyze the topics discussed in documents.'},
#             {'role': 'user', 'content': prompt}
#         ]
#         for i in range(self.max_retry):
#             response = self.client.chat.completions.create(model=self.prompt_model_id, messages=messages, temperature=self.temperature)
#             output = response.choices[0].message.content
#             cost = response.usage.prompt_tokens * api_price[self.prompt_model_id]['input'] + response.usage.completion_tokens * api_price[self.prompt_model_id]['output']
#             try:
#                 result = json.loads(output)
#                 return result, cost
#             except Exception:
#                 continue
#         return
#
#     def eval_self_consistency(self, output_path):
#         eval_result = {}  # {}
#         with tqdm(total=len(self.topics) * self.ann_sample_per_cluster) as pbar:
#             for topic_id in self.topics:
#                 cluster_label = self.topics[topic_id].label
#                 for doc_id in self.topics[topic_id].sampled_docs_ann:
#                     doc_id = int(doc_id)
#                     # print('here ${}$'.format(int(eval_doc_id)), type(self.dataset))
#                     try:
#                         doc_text = self.dataset[int(doc_id)]['text']  # ValueError: invalid literal for int() with base 10: 'IX Corps ( United States )'
#                         if doc_id not in eval_result:
#                             eval_result[doc_id] = {}
#                         # cluster consistency
#                         clu_res, cost = self.__eval_single_doc__(doc_text=doc_text, labels='Topic ID 0: {}'.format(cluster_label))  # {topic_id: 0/1}
#                         eval_result[doc_id][cluster_label] = {'match': clu_res['0'], 'topic_level': 'cluster'}
#                         # individual consistency
#                         doc_labels = [item['topic_name'] for item in self.documents[doc_id].topic_ann]
#                         ind_res, cost = self.__eval_single_doc__(doc_text=doc_text, labels='\n'.join(['Topic ID {}: {}'.format(i, t) for i, t in enumerate(doc_labels)]))
#                         # print('topic output', ind_res)
#                         for doc_tid in range(len(doc_labels)):
#                             eval_result[doc_id][doc_labels[doc_tid]] = {'match': ind_res[str(doc_tid)], 'topic_level': 'document'}
#                         # print('One round', eval_result)
#                         # exit(1)
#                     except Exception:
#                         continue
#                     pbar.update(1)
#
#         # document-level consistency
#         doc_match_cnt = 0
#         cluster_match_cnt = 0
#
#         for doc_id in eval_result:
#             for topic in eval_result[doc_id]:
#                 if eval_result[doc_id][topic]['topic_level'] == 'document':
#                     doc_match_cnt += eval_result[doc_id][topic]['match']
#                 else:
#                     cluster_match_cnt += eval_result[doc_id][topic]['match']
#
#         cluster_consistency = cluster_match_cnt / len(eval_result)
#         doc_consistency = doc_match_cnt / sum([len(eval_result[doc_id]) - 1 for doc_id in eval_result])
#         print('cluster consistency {}, doc consistency {}'.format(cluster_consistency, doc_consistency))
#         with open(output_path, 'w') as fp_output:
#             json.dump({'cluster_consistency': cluster_consistency, 'doc_consistency': doc_consistency, 'full_eval': eval_result}, fp_output)


class LabelEvaluator:
    def __init__(self, documents, topic2doc_weights, doc2topic_weights, topic_annotations, prompt_template, prompt_model_id, temperature, max_retry, doc_max_length):
        self.documents = documents
        self.topic2doc_weights = topic2doc_weights
        self.doc2topic_weights = doc2topic_weights
        self.topic_annotations = topic_annotations
        self.prompt_template = prompt_template
        self.prompt_model_id = prompt_model_id
        self.temperature = temperature
        self.max_retry = max_retry
        self.doc_max_length = doc_max_length

        self.client = OpenAI(api_key=openai_key)

        self.cost = 0

    def score(self, cons_output, gen_output, dist_output):
        cons_score = []
        for topic_id in cons_output:
            cons_score.append(sum(cons_output[topic_id].values()) / len(cons_output[topic_id]))

        gen_score = []
        for topic_id in gen_output:
            score = 0
            weight = 0
            for doc_id in gen_output[topic_id]:
                weight += self.topic2doc_weights[topic_id][int(doc_id)]
                score += int(gen_output[topic_id][doc_id]) * self.topic2doc_weights[topic_id][int(doc_id)]
            gen_score.append(score / weight)

        dist_score = []
        for i in dist_output:
            all_scores = []
            for j in dist_output[i]:
                score = 0
                weight = 0
                for doc_id in dist_output[i][j]:
                    weight += self.topic2doc_weights[j][int(doc_id)]
                    score += int(dist_output[i][j][doc_id]) * self.topic2doc_weights[j][int(doc_id)]
                all_scores.append(score / weight)
            dist_score.append(sum(all_scores) / len(all_scores))

        print('Consistency All: {}. Average: {}'.format(cons_score, sum(cons_score)/len(cons_score)))
        print('Generalization All: {}. Average: {}'.format(gen_score, sum(gen_score)/len(gen_score)))
        print('Distinctiveness All: {}. Average: {}'.format(dist_score, sum(dist_score)/len(dist_score)))

        return {'consistency': cons_score, 'generalization': gen_score, 'distinctiveness': dist_score}

    def consistency(self, chunk=10):
        eval_result = {}
        for topic_id, ann in tqdm(self.topic_annotations.items(), desc='Consistency'):
            documents = []
            for doc_id in ann['pos_docs']:
                documents.append('Document ID: {}\nDocument: {}'.format(doc_id, truncate(self.documents[doc_id]['text'], max_length=256)))
            doc_lists = [documents[i:i + chunk] for i in range(0, len(documents), chunk)]
            # for doc_id in self.sampled_docs[int(topic_id)]['neg']:
            #     documents.append('Document ID: {}\nDocument: {}'.format(doc_id, truncate(self.documents[doc_id]['text'], max_length=256)))
            eval_result[int(topic_id)] = {}
            for doc_lst in doc_lists:
                documents = '\n\n'.join(doc_lst)
                prompt = self.prompt_template.substitute(documents=documents, topic=f"Topic: {ann['topic']['name']}\nDescription: {ann['topic']['description']}")
                result = self.call_api(prompt, len(ann['pos_docs']))
                eval_result[int(topic_id)].update(result)

        return eval_result

    def generalization(self, k, chunk=10):
        eval_result = {}
        with tqdm(total=k * len(self.topic_annotations), desc='Generalization') as pbar:
            for topic_id, ann in self.topic_annotations.items():
                topic_labels = np.argmax(self.doc2topic_weights, axis=1)
                docs_in_cluster = np.where(topic_labels == int(topic_id))[0]
                eval_result[int(topic_id)] = {}
                for i in range(k // chunk):
                    doc_ids = []
                    while len(doc_ids) < chunk:
                        doc_id = random.choice(docs_in_cluster)
                        if doc_id in doc_ids or doc_id in ann['pos_docs']:
                            continue
                        doc_ids.append(doc_id)
                    documents = [f'Document ID: {doc_id}\nDocument: {truncate(self.documents[int(doc_id)]["text"], 256)}' for doc_id in doc_ids]
                    # print('here', len(documents))
                    documents = '\n\n'.join(documents)
                    prompt = self.prompt_template.substitute(documents=documents, topic=f"Topic: {ann['topic']['name']}\nDescription: {ann['topic']['description']}")
                    result = self.call_api(prompt, chunk)
                    # print('output len {}, chunk {}'.format(len(result), chunk))
                    # print(result)
                    # exit(1)
                    eval_result[int(topic_id)] = result
                    pbar.update(chunk)
        return eval_result

    def distinctiveness(self, k):
        eval_result = {}
        with tqdm(total=len(self.topic_annotations) * (len(self.topic_annotations) - 1)) as pbar:
            for i in self.topic_annotations:
                i = int(i)
                eval_result[i] = {}
                for j in self.topic_annotations:
                    j = int(j)
                    if i == j:
                        continue
                    topic_labels = np.argmax(self.doc2topic_weights, axis=1)
                    docs_outside_cluster = np.where(topic_labels == int(j))[0]
                    doc_ids = np.random.choice(docs_outside_cluster, size=k, replace=False)
                    documents = '\n\n'.join([f'Document ID: {doc_id}\nDocument: {truncate(self.documents[int(doc_id)]["text"], 256)}' for doc_id in doc_ids])
                    ann = self.topic_annotations[str(i)]
                    result = self.call_api(self.prompt_template.substitute(documents=documents, topic=f"Topic: {ann['topic']['name']}\nDescription: {ann['topic']['description']}"), k)
                    eval_result[i][j] = result
                    pbar.update(1)
        return eval_result

    def call_api(self, prompt, k):
        messages = [
            {'role': 'system', 'content': 'You are a content analyst that helps me analyze the topics discussed in documents.'},
            {'role': 'user', 'content': prompt}
        ]
        for i in range(self.max_retry):
            response = self.client.chat.completions.create(model=self.prompt_model_id, messages=messages, temperature=self.temperature)
            output = response.choices[0].message.content
            self.cost += response.usage.prompt_tokens * api_price[self.prompt_model_id]['input'] + response.usage.completion_tokens * api_price[self.prompt_model_id]['output']
            try:
                result = json.loads(output)
                # print('here', result)
                # assert len(result) == k, 'k {}, output {}'.format(k, len(result))
                return result
            except Exception as e:
                traceback.print_exc()
                # print(prompt)
                continue
        return


if __name__ == '__main__':
    set_logging(None)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)                      # [wikitext, bill]
    parser.add_argument('--saved_weights', type=str)
    parser.add_argument('--topic_annotations', type=Path)
    parser.add_argument('--prompt_model_id', type=str, default='gpt-4o-mini')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--doc_max_length', type=int, default=500)
    parser.add_argument('--gen_k', type=int, default=50)
    parser.add_argument('--dist_k', type=int, default=5)
    parser.add_argument('--max_retry', type=int, default=3)

    args = parser.parse_args()

    # dataset
    assert args.dataset in ['wikitext', 'bill']

    if args.dataset == 'wikitext':
        dataset = WikitextDataset(data_file=local_config['data']['wikitext'])
    else:
        dataset = BillDataset(data_file=local_config['data']['bill'])

    # topic label result
    with open(args.topic_annotations, 'r') as fp:
        annotations = json.load(fp)

    # saved weights
    with open(args.saved_weights, 'r') as fp:
        saved_weights = json.load(fp)

    evaluator = LabelEvaluator(
        documents=dataset,
        topic2doc_weights=saved_weights['topic2doc_dist'],
        doc2topic_weights=saved_weights['doc2topic_dist'],
        topic_annotations=annotations,
        prompt_template=prompt_eval,
        prompt_model_id=args.prompt_model_id,
        temperature=args.temperature,
        max_retry=args.max_retry,
        doc_max_length=args.doc_max_length,
    )
    print('Evaluating {}'.format(args.topic_annotations))
    generalization_output = evaluator.generalization(k=args.gen_k)
    consistency_output = evaluator.consistency()
    distinctiveness_output = evaluator.distinctiveness(k=args.dist_k)
    scores = evaluator.score(consistency_output, generalization_output, distinctiveness_output)
    #
    with open(os.path.join('output/eval_result', args.dataset, os.path.split(args.topic_annotations)[-1]), 'w') as fp_out:
        json.dump({'consistency': consistency_output, 'generalization': generalization_output, 'distinctiveness': distinctiveness_output, 'scores': scores}, fp_out)

