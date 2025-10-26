import os
import json
import logging
from tqdm import tqdm
from src.label_topic import TopicAnnotator
from src.config import local_config, openai_key
from data.dataset import BillDataset, WikitextDataset
from src.utils import set_logging, truncate
from string import Template 

annotate_prompt_simple = Template("""
You are given a set of documents that belong to the same cluster. Your task is to identify their common topic. The topic must be:
- Broad enough to cover all provided documents and similar ones in the cluster
- Specific enough to avoid including documents from unrelated clusters

Documents from the target cluster:
$pos_docs

Documents from other clusters (optional):
$neg_docs

Follow these rules strictly:
1. Output only a single JSON object.
2. Do NOT include any text outside the JSON (no comments, no code blocks, no explanations).
3. All keys and all string values must be double-quoted.
4. No angle brackets (< >) or triple quotes inside the output.
5. The JSON must be syntactically valid and parsable by json.loads().

Your JSON object must exactly follow this structure:

{
  "name": "Topic name (2-5 words)",
  "description": "One sentence describing what these documents are about.",
  "evidence": {
    "doc_id_1": ["Short evidence snippet 1", "Short evidence snippet 2"],
    "doc_id_2": ["Short snippet"]
  }
}

Ensure:
- Curly braces are balanced
- No trailing commas
- Evidence snippets should be short text phrases from the documents

Output ONLY this JSON. No markdown formatting, no surrounding quotes.
""")

def is_valid_annotation(topic):
    # Checks if the annotation is a dict with required fields and not an error
    if not isinstance(topic, dict):
        return False
    if topic.get("name") == "ERROR":
        return False
    if not all(k in topic for k in ("name", "description", "evidence")):
        return False
    return True

def fix_annotation_file(filepath, dataset, annotator, max_retry=10, stats=None):
    with open(filepath, "r") as f:
        data = json.load(f)
    changed = False
    for topic_id, entry in tqdm(data.items(), desc=f"Checking {os.path.basename(filepath)}"):
        if stats is not None:
            stats['total'] += 1
        topic = entry["topic"]
        if not is_valid_annotation(topic):
            if stats is not None:
                stats['invalid'] += 1
            logging.warning(f"Invalid annotation for topic {topic_id} in {filepath}, retrying...")
            # Re-annotate this topic
            docs = {"pos": entry["pos_docs"], "neg": entry["neg_docs"]}
            for attempt in range(max_retry):
                # Build prompt as in label_topic.py
                pos_docs_texts = []
                for i, doc_id in enumerate(docs['pos']):
                    try:
                        doc_idx = int(doc_id)
                        doc = annotator.documents[doc_idx]
                        pos_docs_texts.append(f"Document {i+1}: {truncate(doc['text'], annotator.max_doc_length)}")
                    except Exception:
                        continue
                pos_documents = '\n'.join(pos_docs_texts)
                neg_docs_texts = []
                for i, doc_id in enumerate(docs['neg']):
                    try:
                        doc_idx = int(doc_id)
                        doc = annotator.documents[doc_idx]
                        neg_docs_texts.append(f"Document {i+1}: {truncate(doc['text'], annotator.max_doc_length)}")
                    except Exception:
                        continue
                neg_documents = '\n'.join(neg_docs_texts)
                if not neg_documents:
                    neg_documents = 'No documents'
                prompt = annotator.prompt_template.substitute(pos_docs=pos_documents, neg_docs=neg_documents)
                messages = [
                    {'role': 'system', 'content': 'You are a help content analyst that helps me analyze the topics discussed in documents.'},
                    {'role': 'user', 'content': prompt}
                ]
                try:
                    response = annotator.client.chat.completions.create(
                        model=annotator.model_id,
                        messages=messages,
                        temperature=annotator.temperature
                    )
                    result = response.choices[0].message.content
                    # Try to extract JSON if needed
                    try:
                        result_json = json.loads(result)
                    except Exception:
                        import re
                        match = re.search(r"\{.*\}", result, re.DOTALL)
                        if match:
                            result_json = json.loads(match.group(0))
                        else:
                            result_json = {"name": "ERROR", "description": "Annotation failed", "evidence": {}}
                    if is_valid_annotation(result_json):
                        entry["topic"] = result_json
                        changed = True
                        if stats is not None:
                            stats['fixed'] += 1
                        logging.info(f"Fixed topic {topic_id} in {filepath}")
                        break
                except Exception as e:
                    logging.warning(f"Retry {attempt+1}/{max_retry} failed for topic {topic_id}: {e}")
            else:
                logging.error(f"Failed to fix topic {topic_id} in {filepath} after {max_retry} attempts.")
    if changed:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved corrected file: {filepath}")

if __name__ == "__main__":
    import argparse
    set_logging(log_file=None)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_model_id', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--max_retry', type=int, default=5)
    args = parser.parse_args()

    stats = {'total': 0, 'invalid': 0, 'fixed': 0}
    for dataset_name in ["wikitext", "bill"]:
        if dataset_name == "wikitext":
            dataset = WikitextDataset(data_file=local_config['data']['wikitext'])
        else:
            dataset = BillDataset(data_file=local_config['data']['bill'])
        annotator = TopicAnnotator(
            model_id=args.prompt_model_id,
            temperature=args.temperature,
            prompt_template=annotate_prompt_simple,
            max_retry=args.max_retry,
            documents=dataset
        )
        ann_dir = f"output/annotations/{dataset_name}"
        for fname in os.listdir(ann_dir):
            if fname.endswith(".json"):
                fix_annotation_file(os.path.join(ann_dir, fname), dataset, annotator, max_retry=args.max_retry, stats=stats)

    print(f"\nSummary:")
    print(f"Total topics checked: {stats['total']}")
    print(f"Invalid topics detected: {stats['invalid']}")
    print(f"Successfully fixed: {stats['fixed']}")