"""
Topic/Clustering model
"""
import json
import logging
import argparse
import sys
import os
import tomotopy as tp
import numpy as np
from data.dataset import WikitextDataset, BillDataset
from src.utils import set_logging
from src.config import local_config
from collections import OrderedDict
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
import nltk
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import CountVectorizer
from contextualized_topic_models.utils.data_preparation import CTMDataset


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

def train_ctm(corpus, num_topics, save_path, embeddings_path, vocab_path, num_epochs=50, batch_size=64, lr=5e-3):
    logging.info('Train CTM model')
    
    # Load embeddings
    table = pq.read_table(embeddings_path)
    df_embeddings = table.to_pandas()
    embeddings_list = df_embeddings['embeddings'].apply(lambda x: [float(num) for num in x.split()]).tolist()
    embeddings = np.array(embeddings_list)
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_dict = json.load(f)
    vocab = [None] * len(vocab_dict)
    for word, idx in vocab_dict.items():
        vocab[idx] = word
    
    # Create BoW
    preprocessed_texts = [' '.join(doc['tokenized_text']) for doc in corpus]
    vectorizer = CountVectorizer(vocabulary=vocab_dict, lowercase=False, token_pattern=r'(?u)\b\w+\b')
    bow_matrix = vectorizer.fit_transform(preprocessed_texts)
    
    # Create dataset
    training_dataset = CTMDataset(
        X_contextual=embeddings,
        X_bow=bow_matrix,
        idx2token=vocab
    )
    
    # Train model
    ctm = CombinedTM(
        bow_size=len(vocab),
        contextual_size=embeddings.shape[1],
        n_components=num_topics,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr
    )
    
    logging.info('{} docs, {} topics, {} epochs'.format(len(corpus), num_topics, num_epochs))
    ctm.fit(training_dataset)
    
    # Extract outputs
    topic2word = ctm.get_topic_word_distribution()  # [T, V]
    doc2topic = ctm.get_doc_topic_distribution(training_dataset)
    topic2doc = (doc2topic / doc2topic.sum(axis=0, keepdims=True)).T
    
    with open(save_path, 'w') as fp:
        json.dump({
            'topic2word_dist': topic2word.tolist(),
            'doc2topic_dist': doc2topic.tolist(),
            'topic2doc_dist': topic2doc.tolist(),
            'vocab': vocab
        }, fp)
    
    logging.info(f'Final loss: {ctm.best_loss_train:.2f}')


def train_bertopic(corpus, num_topics, save_path, embeddings_path, vocab_path):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic.vectorizers import ClassTfidfTransformer
    
    logging.info('Train BERTopic model')
    
    # Load embeddings
    table = pq.read_table(embeddings_path)
    df_embeddings = table.to_pandas()
    embeddings_list = df_embeddings['embeddings'].apply(lambda x: [float(num) for num in x.split()]).tolist()
    embeddings = np.array(embeddings_list)
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_dict = json.load(f)
    vocab = [None] * len(vocab_dict)
    for word, idx in vocab_dict.items():
        vocab[idx] = word
    
    # Prepare texts
    preprocessed_texts = [' '.join(doc['tokenized_text']) for doc in corpus]
    
    # Configure models
    vectorizer_model = CountVectorizer(vocabulary=vocab_dict, lowercase=False, token_pattern=r'(?u)\b\w+\b')
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    # Train BERTopic
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        nr_topics=num_topics + 1,  # +1 for outlier topic
        calculate_probabilities=True,
        verbose=True
    )
    
    logging.info('{} docs, {} topics'.format(len(corpus), num_topics))
    topics, probs = topic_model.fit_transform(preprocessed_texts, embeddings)
    
    # Get approximate distribution
    thetas_approx = topic_model.approximate_distribution(preprocessed_texts, use_embedding_model=False)
    
    # Extract topic-word matrix
    topic2word_raw = topic_model.c_tf_idf_.toarray()
    if topic2word_raw.shape[0] == num_topics + 1:
        topic2word = topic2word_raw[1:, :]  # Drop outlier row
    else:
        topic2word = topic2word_raw
    topic2word = topic2word / (topic2word.sum(axis=1, keepdims=True) + 1e-10)
    
    # Topic-doc distribution
    topic2doc = (thetas_approx[0] / (thetas_approx[0].sum(axis=0, keepdims=True) + 1e-10)).T
    
    with open(save_path, 'w') as fp:
        json.dump({
            'topic2word_dist': topic2word.tolist(),
            'doc2topic_dist': thetas_approx[0].tolist(),
            'doc2topic_probs': probs.tolist(),
            'topic2doc_dist': topic2doc.tolist(),
            'vocab': vocab
        }, fp)
    
    logging.info(f'Outliers: {sum(1 for t in topics if t == -1)} ({sum(1 for t in topics if t == -1)/len(topics)*100:.1f}%)')


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='eval')   # [train, eval]
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--num_topics', type=int, default=25)
    parser.add_argument('--model_type', type=str, default='lda')
    parser.add_argument('--max_iterations', type=int)
    parser.add_argument('--num_epochs', type=int, default=50)

    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    log_file = f'logs/{args.model_type}_{args.dataset}_{args.num_topics}.log'
    set_logging(log_file=log_file)

    if args.dataset == 'wikitext':
        documents = WikitextDataset(data_file=local_config['data'][args.dataset])
    else:
        documents = BillDataset(data_file=local_config['data'][args.dataset])

    if args.task == 'train':
        if args.model_type == 'lda':
            train_lda(corpus=documents, num_topics=args.num_topics, max_iterations=args.max_iterations, save_path=f'output/topic_models/{args.model_type}_{args.dataset}_{args.num_topics}.json')
        
        elif args.model_type == 'ctm':
            train_ctm(corpus=documents, 
                      num_topics=args.num_topics, 
                      save_path=f'output/topic_models/{args.model_type}_{args.dataset}_{args.num_topics}.json', 
                      embeddings_path=local_config['embeddings'][args.dataset], 
                      vocab_path=local_config['vocab'][args.dataset], 
                      num_epochs=args.num_epochs)
        
        elif args.model_type == 'bertopic':
            print('Training BERTopic model...')
            train_bertopic(corpus=documents, 
                        num_topics=args.num_topics, 
                        save_path=f'output/topic_models/{args.model_type}_{args.dataset}_{args.num_topics}.json', 
                        embeddings_path=local_config['embeddings'][args.dataset], 
                        vocab_path=local_config['vocab'][args.dataset])
            

    elif args.task == 'eval':
        with open(f'output/topic_models/{args.model_type}_{args.dataset}_{args.num_topics}.json', 'r') as fp_output:
            topic_model_output = json.load(fp_output)
        evaluator = TopicModelEvaluator(dataset=documents, output=topic_model_output)
        result = evaluator.purity()
        print('Number of topics {}, purity scores {}'.format(args.num_topics, result))
