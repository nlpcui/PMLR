import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
import numpy as np
import argparse
import logging
from data.dataset import Features


def plot_recall(result_files, save_path):

    # markers = ['o', 's', '^', 'D']
    x = np.arange(0, 10)
    print(result_files)
    for idx, (na, rf) in enumerate(result_files):
        with open(rf, 'r') as fp:
            result = json.load(fp)

        deciles = [[] for i in range(10)]
        std = [0 for i in range(10)]
        for feature_id in result:
            labels = list(result[feature_id]['match']['pos'].values())
            for i in range(10):
                deciles[i].extend(labels[i * 10:(i + 1) * 10])
        for i in range(10):
            deciles[i] = sum(deciles[i]) / len(deciles[i])
        deciles = np.array(deciles)
        plt.plot(x, deciles, label=na, marker='o')

    ax = plt.gca()
    ax.set_xticks(x)
    ax.tick_params(
        axis='x',
        direction='out',  # 向外
        length=4,
        width=1
    )
    plt.yticks(np.arange(0, 1.01, 0.2))
    plt.xticks(x, labels=['']*10)
    plt.ylabel('Recall', fontsize=12)
    plt.xlabel('Percentile', fontsize=12)
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    # precision = []
    # deciles = [[] for i in range(10)]
    # for feature_id in result:
    #     labels = list(result[feature_id]['match']['pos'].values())
    #     for i in range(10):
    #         deciles[i].extend(labels[i*10:(i+1)*10])
    #
    #     precision.append(sum(result[feature_id]['match']['pos'].values()) / (sum(result[feature_id]['match']['pos'].values()) + sum(result[feature_id]['match']['neg'].values())) )
    #
    # for i in range(10):
    #     deciles[i] = sum(deciles[i]) / len(deciles[i])
    #
    # print('deciles recall', deciles, 'mean recall', np.mean(deciles), 'precision: ', np.mean(precision))


def print_results(result_files, features, weighted):

    table = PrettyTable()
    table.field_names = ['Method', 'Precision', 'Recall', 'F1']
    for idx, (na, rf) in enumerate(result_files):
        with open(rf, 'r') as fp:
            result = json.load(fp)
        precisions = []
        recalls = []
        f1_scores = []
        for feature_id in result:
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for data_id, value in result[feature_id]['match']['pos'].items():
                data_id = int(data_id)
                if value == 1:
                    tp += np.max(features.data2feature_weights[data_id]) if weighted else 1
                else:
                    fn += np.max(features.data2feature_weights[data_id]) if weighted else 1
            for data_id, value in result[feature_id]['match']['neg'].items():
                data_id = int(data_id)
                if value == 1:
                    fp += np.max(features.data2feature_weights[data_id]) if weighted else 1
                else:
                    tn += np.max(features.data2feature_weights[data_id]) if weighted else 1

            precisions.append(tp/(tp+fp+1e-5))
            recalls.append(tp/(tp+fn+1e-5))
            f1_scores.append(2*precisions[-1]*recalls[-1] / (precisions[-1] + recalls[-1] + 1e-5))

        precision_mean = round(float(np.mean(precisions)), 3)
        precision_std = round(float(np.std(precisions)), 3)

        recall_mean = round(float(np.mean(recalls)), 3)
        recall_std = round(float(np.std(recalls)), 3)

        f1_score_mean = round(float(np.mean(f1_scores)), 3)
        f1_score_std = round(float(np.std(f1_scores)), 3)

        table.add_row([
            na,
            f'{precision_mean} ({precision_std})',
            f'{recall_mean}, ({recall_std})',
            f'{f1_score_mean}, ({f1_score_std})'
        ])

    print(table)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--model', type=str, default='sae')
    parser.add_argument('--num_topics', type=int, default=256)
    parser.add_argument('--saved_features', type=str, default='output/topic_models/sae_wikitext_256.json')
    parser.add_argument('--methods', type=lambda x: x.split(','), default='random,stratified,topk,weighted,dpp_0.05')
    parser.add_argument('--print_recall', type=bool, )
    parser.add_argument('--print_all', type=bool, )
    parser.add_argument('--feature_type', type=str, default='sae')
    args = parser.parse_args()

    for num_k in [5, 10, 20]:
        plot_recall(
            result_files=[
                ('random', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_random_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                ('stratified', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_stratified_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                ('topk', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_topk_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                ('weighted', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_weighted_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                # ('dpp_0.1', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_dpp_0.1_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                ('dpp_0.05', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_dpp_0.05_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                # ('dpp_0.01', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_dpp_0.01_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                # ('dpp', f'output/eval_result/{args.dataset}/lda{args.num_topics}_label_dpp_0.05_pos{num_k}_neg0_eval_stratified_pos100_neg100.json')
            ],
            save_path=f'output/plot/{args.dataset}_{args.model}_recall_{num_k}.png'
        )

    logging.info('Load features from {}'.format(args.saved_features))
    saved_features = Features.load(feature_file=args.saved_features, feature_type=args.feature_type)
    #
    for num_k in [5, 10, 20]:
        print_results(
            result_files=[
                ('random', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_random_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                ('stratified', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_stratified_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                ('topk', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_topk_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                ('weighted', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_weighted_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                ('dpp_0.1', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_dpp_0.1_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                ('dpp_0.01', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_dpp_0.01_pos{num_k}_neg0_eval_stratified_pos100_neg100.json'),
                ('dpp_0.05', f'output/eval_result/{args.dataset}/{args.model}{args.num_topics}_label_dpp_0.05_pos{num_k}_neg0_eval_stratified_pos100_neg100.json')
            ],
            features=saved_features,
            weighted=True
        )
