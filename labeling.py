import json
import argparse
import logging

import numpy as np

from config import local_config, openai_key, api_price
from openai import OpenAI
from data.dataset import BillDataset, WikitextDataset
from sampling import Selector
from tqdm import tqdm
from collections import OrderedDict
from utils import set_logging, truncate
from string import Template

annotate_prompt = \
Template("""
You are given a collection of documents that belong to the same semantic cluster, along with a list of keywords of the cluster. Your task is to analyze these documents and identify the common topic they share by following the steps below:

Step 1: For each document, identify all major topics discussed. For each topic, provide: (1) A concise topic name (approximately 2–5 words), and (2) One or more evidence snippets extracted from the document that support the existence of the topic.
Step 2: Identify the smallest common topic based on the document-level topics. The topic should be broad enough to encompass all the documents, yet specific enough to avoid overgeneralization or inclusion of information not present in the documents.

For the resulting common topic, provide:
(a) A topic name;
(b) A one-sentence description;
(c) One or more evidence snippets from each document that support the common topic;

Your output must be provided in the following json format:
{
  "document_topics": {
    "<document_id_1>": [
      {
        "topic_name": "<topic name 1>",
        "evidence": ["<evidence snippet 1>", "<evidence snippet 2>"]
      },
      {
        "topic_name": "<topic name 2>",
        "evidence": ["<evidence snippet>", ...]
      }
    ],
    "<document_id_2>": [
      ...
    ]
  },
  "common_topic": {
    "topic_name": "<common topic name>",
    "description": "<one-sentence description>",
    "evidence": {
      "<document_id_1>": ["<evidence snippet 1>", "<evidence snippet 2>"],
      "<document_id_2>": ["<evidence snippet>"]
    }
  }
}

Below is the set of documents:

$documents

Below is the keywords:

$keywords

Directly provide the JSON output without any other output including the format code like ```json```, etc. 
Your json output:""")


class TopicAnnotator:
    def __init__(self, model_id, temperature, max_doc_length=500, prompt_template=annotate_prompt, num_keywords=10, max_retry=3):
        self.temperature = temperature
        self.client = OpenAI(api_key=openai_key)
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.max_retry = max_retry
        self.max_doc_length = max_doc_length
        self.num_keywords = num_keywords

    def __annotate_single(self, docs, keywords):
        doc_id_text = [(doc['id'], doc['text']) for doc in docs]
        documents = []
        for doc_id, doc_text in doc_id_text:
            documents.append('Document ID: {doc_id}\nContent: {doc_text}'.format(
                doc_id=doc_id, doc_text=truncate(doc_text, self.max_doc_length)
            ))
        documents = '\n\n'.join(documents)
        prompt = self.prompt_template.substitute(documents=documents, keywords=','.join(keywords))
        # print('prompt', prompt)
        # exit(1)

        messages = [{'role': 'system', 'content': 'You are a help content analyst that helps me analyze the topics discussed in documents.'}, {'role': 'user', 'content': prompt}]

        output = {
            'prompt_tokens': 0, 'completion_tokens': 0,
        }
        for i in range(self.max_retry):
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature
            )
            try:
                result = response.choices[0].message.content
                result = json.loads(result)
                output['prompt_tokens'] += response.usage.prompt_tokens
                output['completion_tokens'] += response.usage.completion_tokens
                output['annotation'] = result
                return output
            except Exception:
                continue

        return

    def annotate(self, topic2doc_weights, topic2word_weights, vocab, doc_selector, save_path, samples_per_cluster, sel_func):
        # clusters: scores []
        annotations = {}
        total_input_tokens = 0
        total_output_tokens = 0
        with tqdm(total=len(topic2doc_weights) * samples_per_cluster) as pbar:
            for cid, weights in enumerate(topic2doc_weights):
                # sample documents
                selected_docs = doc_selector.select(sel_func, weights, k=samples_per_cluster)
                # annotation
                keyword_ids = np.argsort(-np.array(topic2word_weights), axis=-1)[cid, :self.num_keywords]
                keywords = [vocab[kid] for kid in keyword_ids]
                result = self.__annotate_single(selected_docs, keywords)
                if not result:
                    logging.error('Topic {} failed!'.format(cid))
                print('Topic {}, keywords {}, annotated label {}, description {}'.format(cid, keywords, result['annotation']['common_topic']['topic_name'], result['annotation']['common_topic']['description']))
                # exit(1)
                annotations[cid] = result['annotation']
                total_input_tokens += result['prompt_tokens']
                total_output_tokens += result['completion_tokens']
                pbar.update(samples_per_cluster)
                pbar.set_postfix(OrderedDict({
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens,
                    'cost': api_price[self.model_id]['input'] * total_input_tokens + api_price[self.model_id]['output'] * total_output_tokens
                }))

        with open(save_path, 'w') as fp_out:
            json.dump(annotations, fp_out)


if __name__ == "__main__":
    set_logging(log_file=None)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)              # [wikitext, bill]
    parser.add_argument('--sel_func', type=str,)            # [random, top_k, ]
    parser.add_argument('--k', type=int, )                  # number of samples per cluster
    parser.add_argument('--output_path', type=str, )        # save_dir
    parser.add_argument('--prompt_model_id', type=str)      # GPT model
    parser.add_argument('--temperature', type=float)        #
    parser.add_argument('--saved_weights', type=str)

    args = parser.parse_args()

    if args.dataset == 'wikitext':
        dataset = WikitextDataset(data_file=local_config['data']['wikitext'])
    else:
        dataset = BillDataset(data_file=local_config['data']['bill'])

    with open(args.saved_weights, 'r') as fp:
        saved_weights = json.load(fp)

    logging.info('loading saved weights from {}, {} topics, {} docs.'.format(args.saved_weights, len(saved_weights['topic2doc_dist']), len(saved_weights['topic2doc_dist'][0])))

    selector = Selector(dataset=dataset)

    annotator = TopicAnnotator(
        model_id=args.prompt_model_id,
        temperature=args.temperature,
        prompt_template=annotate_prompt,
        max_retry=3,
    )

    annotator.annotate(
        topic2doc_weights=saved_weights['topic2doc_dist'],
        topic2word_weights=saved_weights['topic2word_dist'],
        vocab=saved_weights['vocab'],
        doc_selector=selector,
        save_path=args.output_path,
        samples_per_cluster=args.k,
        sel_func=args.sel_func
    )


