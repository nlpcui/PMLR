import json
import argparse
import logging
import os
from tqdm import tqdm
from collections import OrderedDict
from string import Template
from src.config import local_config, openai_key, api_price
from openai import OpenAI
from data.dataset import BillDataset, WikitextDataset
from src.utils import set_logging, truncate

annotate_prompt_simple = Template("""
You are given a set of documents that belong to the same cluster. Your task is to identify their common topic. The topic should be broad enough to cover all provided documents and other potential documents in the same cluster, and specific enough to avoid overgeneralization to documents from other clusters. If documents from other clusters are provided, you may use them as references for contrast.

Documents from the target cluster:
$pos_docs

Documents from other clusters (optional):
$neg_docs

Please output your result as a single JSON object with the following fields:
{"name": "topic_name (2-5 words)", 
"description": "one-sentence description of the topic",
"evidence": {
      "<document_id_1>": ["<evidence snippet 1>", "<evidence snippet 2>"],
      "<document_id_2>": ["<evidence snippet>"]
    }}

Only output the JSON string and do not include any additional text, explanations, or code formatting (such as json or markdown)!
""")

class TopicAnnotator:
    def __init__(self, documents, model_id, temperature, prompt_template, max_doc_length=256, max_retry=3):
        self.temperature = temperature
        self.client = OpenAI(api_key=openai_key)
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.max_retry = max_retry
        self.max_doc_length = max_doc_length
        self.documents = documents

    def annotate(self, sampled_docs):
        total_input_tokens = 0
        total_output_tokens = 0
        topic_labels = {}

        with tqdm(total=len(sampled_docs)) as pbar:
            for topic_id, docs in sampled_docs.items():
                # Safely build pos_documents
                pos_docs_texts = []
                for i, doc_id in enumerate(docs['pos']):
                    try:
                        doc_idx = int(doc_id)
                        doc = self.documents[doc_idx]
                        pos_docs_texts.append(f"Document {i+1}: {truncate(doc['text'], self.max_doc_length)}")
                    except (IndexError, KeyError):
                        logging.warning(f"doc_id {doc_id} out of range for dataset of size {len(self.documents)}. Skipping.")
                        continue
                pos_documents = '\n'.join(pos_docs_texts)

                # Safely build neg_documents
                neg_docs_texts = []
                for i, doc_id in enumerate(docs['neg']):
                    try:
                        doc_idx = int(doc_id)
                        doc = self.documents[doc_idx]
                        neg_docs_texts.append(f"Document {i+1}: {truncate(doc['text'], self.max_doc_length)}")
                    except (IndexError, KeyError):
                        logging.warning(f"doc_id {doc_id} out of range for dataset of size {len(self.documents)}. Skipping.")
                        continue
                neg_documents = '\n'.join(neg_docs_texts)
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
                    except Exception as e:
                        logging.warning(f"Retry {i+1}/{self.max_retry} failed for topic {topic_id}: {e}")
                        continue
                if not isinstance(result, dict):
                    logging.error(f"Failed to parse LLM response for topic {topic_id}. Raw result: {result}")
                    result = {"name": "ERROR", "description": "Annotation failed", "evidence": {}}
                logging.info(f"Annotated topic {topic_id}: {result.get('name', 'N/A')}")
                pbar.update(1)
                pbar.set_postfix(OrderedDict({
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens,
                    'cost': api_price[self.model_id]['input'] * total_input_tokens + api_price[self.model_id]['output'] * total_output_tokens
                }))
                topic_labels[topic_id] = {'pos_docs': docs['pos'], 'neg_docs': docs['neg'], 'topic': result}
        return topic_labels
    
if __name__ == "__main__":
    set_logging(log_file=None)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)              # [wikitext, bill]
    parser.add_argument('--prompt_model_id', type=str, required=True)      # GPT model
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--samples', type=str, required=True)              # Path to a single sampled doc file

    args = parser.parse_args()
    dataset = WikitextDataset(data_file=local_config['data']['wikitext']) if args.dataset == 'wikitext' else BillDataset(data_file=local_config['data']['bill'])

    annotator = TopicAnnotator(
        model_id=args.prompt_model_id,
        temperature=args.temperature,
        prompt_template=annotate_prompt_simple,
        max_retry=3,
        documents=dataset
    )

    # Prepare output path
    os.makedirs(f'output/annotations/{args.dataset}', exist_ok=True)
    output_file = f'output/annotations/{args.dataset}/{os.path.split(args.samples)[-1]}'

    # Skip if already annotated
    if os.path.exists(output_file):
        logging.info(f"Annotation for {output_file} already exists. Skipping.")
        print(f"Annotation for {output_file} already exists. Skipping.")
    else:
        logging.info(f'Labeling docs from {args.samples} for dataset {args.dataset}')
        print(f'Labeling docs from {args.samples} for dataset {args.dataset}')
        with open(args.samples, 'r') as fp_in:
            sampled_documents = json.load(fp_in)
        labels = annotator.annotate(sampled_docs=sampled_documents)
        with open(output_file, 'w') as fp_out:
            json.dump(labels, fp_out)

# if __name__ == "__main__":
#     set_logging(log_file=None)
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str)              # [wikitext, bill]
#     parser.add_argument('--prompt_model_id', type=str)      # GPT model
#     parser.add_argument('--temperature', type=float)        #
#     parser.add_argument('--saved_scores', type=str)
#     parser.add_argument('--samples', type=str)

#     args = parser.parse_args()
#     dataset = WikitextDataset(data_file=local_config['data']['wikitext']) if args.dataset == 'wikitext' else BillDataset(data_file=local_config['data']['bill'])

#     annotator = TopicAnnotator(
#         model_id=args.prompt_model_id,
#         temperature=args.temperature,
#         prompt_template=annotate_prompt_simple,
#         max_retry=3,
#         documents=dataset
#     )
#     all_sampled = []
#     if os.path.isdir(args.samples):
#         for filename in os.listdir(args.samples):
#             if os.path.isfile(os.path.join(args.samples, filename)):
#                 all_sampled.append(os.path.join(args.samples, filename))
#     else:
#         all_sampled.append(args.samples)

#     os.makedirs(f'output/annotations/{args.dataset}', exist_ok=True)

#     for input_file in all_sampled:
#         logging.info(f'Labeling docs from {input_file} for dataset {args.dataset}')
#         print(f'Labeling docs from {input_file} for dataset {args.dataset}')
#         with open(input_file, 'r') as fp_in:
#             sampled_documents = json.load(fp_in)
#         labels = annotator.annotate(sampled_docs=sampled_documents)
#         with open(f'output/annotations/{args.dataset}/{os.path.split(input_file)[-1]}', 'w') as fp_out:
#             json.dump(labels, fp_out)