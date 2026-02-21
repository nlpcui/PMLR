import json
import argparse
import logging
import os
import sys
from tqdm import tqdm
from collections import OrderedDict
from string import Template
from src.config import local_config, api_price
from openai import OpenAI
from data.dataset import BillDataset, WikitextDataset, SAEWikitext, Features
from src.utils import set_logging, truncate
from pathlib import Path

openai_key = os.getenv("OPENAI_API_KEY_Data_Interpretation")

print('openai_key', openai_key)

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


annotate_prompt_simple = Template("""
Given a set of documents that belong to the same cluster, your task is to identify their common topic. The topic should be broad enough to cover all provided documents and other potential documents in the same cluster, and specific enough to avoid overgeneralization to documents from other clusters. If documents from other clusters are provided, you can use them as references for contrast.

Documents from the target cluster:
$pos_docs

Documents from other clusters (optional):
$neg_docs

Please output your result as a single JSON object with the following fields:
{"name": "topic_name (2-5 words)", "description": "one-sentence description of the topic"}

Only output the JSON string and do not include any additional text, explanations, or code formatting (such as json or markdown)!
""")

annotate_prompt_explanation = Template("""
You are a meticulous AI researcher conducting an important investigation into patterns found in language. You will be given a list of text examples and your task is to analyze text and provide an interpretation that thoroughly encapsulates possible patterns found in it.

Text examples:
$EXAMPLES

Output exactly one valid JSON object in the following format:
{"title": "concise pattern title, max 5 words", "description": "one-sentence description of the pattern"}.

Do not output any additional text, explanations, or code formatting (such as json or markdown).
""")


class TopicAnnotator:
    def __init__(self, documents, model_id, temperature, prompt_template, features, min_feature_size, max_doc_length=256, max_retry=3):
        self.temperature = temperature
        self.client = OpenAI(api_key=openai_key)
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.max_retry = max_retry
        self.max_doc_length = max_doc_length
        self.documents = documents
        self.features = features
        self.min_feature_size = min_feature_size

    # def __annotate_single(self, docs, keywords):
    #     doc_id_text = [(doc['id'], doc['text']) for doc in docs]
    #     documents = []
    #     for doc_id, doc_text in doc_id_text:
    #         documents.append('Document ID: {doc_id}\nContent: {doc_text}'.format(
    #             doc_id=doc_id, doc_text=truncate(doc_text, self.max_doc_length)
    #         ))
    #     documents = '\n\n'.join(documents)
    #     prompt = self.prompt_template.substitute(documents=documents, keywords=','.join(keywords))
    #     # print('prompt', prompt)
    #     # exit(1)
    #
    #     messages = [{'role': 'system', 'content': 'You are a help content analyst that helps me analyze the topics discussed in documents.'}, {'role': 'user', 'content': prompt}]
    #
    #     output = {
    #         'prompt_tokens': 0, 'completion_tokens': 0,
    #     }
    #     for i in range(self.max_retry):
    #         response = self.client.chat.completions.create(
    #             model=self.model_id,
    #             messages=messages,
    #             temperature=self.temperature
    #         )
    #         try:
    #             result = response.choices[0].message.content
    #             result = json.loads(result)
    #             output['prompt_tokens'] += response.usage.prompt_tokens
    #             output['completion_tokens'] += response.usage.completion_tokens
    #             output['annotation'] = result
    #             return output
    #         except Exception:
    #             continue
    #
    #     return

    def annotate(self, sampled_docs):
        total_input_tokens = 0
        total_output_tokens = 0

        # doc2topic = {doc_id: int(np.argmax(doc2topic_weights[doc_id])) for doc_id in range(len(doc2topic_weights))}
        # topic2doc = {topic_id: set() for topic_id in range(len(topic2doc_weights))}
        # for doc_id in doc2topic:
        #     topic2doc[doc2topic[doc_id]].add(doc_id)

        skipped_features = {}

        topic_labels = {}

        with tqdm(total=len(sampled_docs)) as pbar:
            for topic_id, docs in sampled_docs.items():
                if len(self.features.label2data[int(topic_id)]) < self.min_feature_size:
                    skipped_features[int(topic_id)] = len(self.features.label2data[int(topic_id)])
                    pbar.update(1)
                    continue
                pos_documents = '\n'.join([f"Document {i+1}: {truncate(self.documents[int(doc_id)]['text'], self.max_doc_length)}" for i, doc_id in enumerate(docs['pos'])])
                neg_documents = '\n'.join([f"Document {i+1}: {truncate(self.documents[int(doc_id)]['text'], self.max_doc_length)}" for i, doc_id in enumerate(docs['neg'])])
                if not neg_documents:
                    neg_documents = 'No documents'
                prompt = self.prompt_template.substitute(pos_docs=pos_documents, neg_docs=neg_documents)

                messages = [
                    {'role': 'system', 'content': 'You are a help content analyst that helps me analyze the topics discussed in documents.'},
                    {'role': 'user', 'content': prompt}
                ]
                result = None
                for i in range(self.max_retry):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_id,
                            messages=messages,
                            temperature=self.temperature
                        )
                        result = response.choices[0].message.content
                        result = json.loads(result)
                        total_input_tokens += response.usage.prompt_tokens
                        total_output_tokens += response.usage.completion_tokens
                        break
                    except Exception:
                        continue
                pbar.update(1)
                pbar.set_postfix(OrderedDict({
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens,
                    'cost': api_price[self.model_id]['input'] * total_input_tokens + api_price[self.model_id]['output'] * total_output_tokens
                }))

                topic_labels[topic_id] = {'pos_docs': docs['pos'], 'neg_docs': docs['neg'], 'topic': result}

        logging.info('Skipped features: {}'.format(skipped_features))

        return topic_labels


if __name__ == "__main__":
    set_logging(log_file=None)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)              # [topic_wikitext, topic_bill]
    parser.add_argument('--max_text_length', type=int, default=256)
    parser.add_argument('--prompt_model_id', type=str)      # GPT model
    parser.add_argument('--temperature', type=float)        #
    parser.add_argument('--saved_scores', type=str)
    parser.add_argument('--samples', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--saved_features', type=Path)
    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--min_feature_size', type=int, default=500)

    args = parser.parse_args()
    if args.dataset == 'topic_wikitext':
        dataset = WikitextDataset(data_file=local_config['data']['topic_wikitext'])
        ann_prompt = annotate_prompt_simple
    elif args.dataset == 'topic_bill':
        dataset = BillDataset(data_file=local_config['data']['topic_bill'])
        ann_prompt = annotate_prompt_simple
    elif args.dataset == 'sae_wikitext':
        dataset = SAEWikitext(data_file=local_config['data']['sae_wikitext'])
        ann_prompt = annotate_prompt_explanation

    saved_features = Features.load(args.saved_features, feature_type=args.feature_type)

    annotator = TopicAnnotator(
        model_id=args.prompt_model_id,
        temperature=args.temperature,
        prompt_template=ann_prompt,
        max_retry=3,
        documents=dataset,
        features=saved_features,
        min_feature_size=args.min_feature_size,
        max_doc_length=args.max_text_length
    )

    with open(args.samples, 'r') as fp_samples:
        sampled_data = json.load(fp_samples)

    labels = annotator.annotate(sampled_docs=sampled_data)
    with open(args.output_file, 'w') as fp_out:
        json.dump(labels, fp_out)

    # all_sampled = []
    # if os.path.isdir(args.samples):
    #     for filename in os.listdir(args.samples):
    #         if os.path.isfile(os.path.join(args.samples, filename)):
    #             all_sampled.append(os.path.join(args.samples, filename))
    # else:
    #     all_sampled.append(args.samples)
    #
    # for input_file in all_sampled:
    #     logging.info('Labeling docs from {}'.format(input_file))
    #     with open(input_file, 'r') as fp_in:
    #         sampled_documents = json.load(fp_in)
    #     labels = annotator.annotate(sampled_docs=sampled_documents)
    #     with open(f'output/topic_labels/{args.dataset}/{os.path.split(input_file)[-1]}', 'w') as fp_out:
    #         json.dump(labels, fp_out)

