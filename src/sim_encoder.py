import argparse
import os
import sys
import json
import torch
from sentence_transformers import SentenceTransformer
from data.dataset import WikitextDataset, BillDataset, SAEWikitext
from src.config import local_config


def pre_trained_embeddings(ids, texts, model_id):
    model = SentenceTransformer(model_id)
    encoded = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return {ids[i]: encoded[i].tolist() for i in range(len(ids))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sae_wikitext')
    parser.add_argument('--model_id', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--job', type=str, default='generate')
    parser.add_argument('--save_dir', type=str, default='output/embeddings')

    args = parser.parse_args()

    if args.dataset == 'topic_wikitext':
        dataset = WikitextDataset(data_file=local_config['data'][args.dataset])
    elif args.dataset == 'topic_bill':
        dataset = BillDataset(data_file=local_config['data'][args.dataset])
    elif args.dataset == 'sae_wikitext':
        dataset = SAEWikitext(data_file=local_config['data'][args.dataset])

    if args.job == 'train':
        pass

    elif args.job == 'generate':
        embeddings = pre_trained_embeddings(ids=dataset.ids, texts=dataset.texts, model_id=args.model_id)

        with open(os.path.join(args.save_dir, f'{args.dataset}_pretrained.json'), 'w') as fp_out:
            json.dump(embeddings, fp_out)
