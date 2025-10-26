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

