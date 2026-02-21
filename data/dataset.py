import json
import sys

import matplotlib.pyplot as plt
import nltk
import numpy as np
from collections import OrderedDict
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from dataclasses import dataclass
en_stopwords = set(stopwords.words('english'))


def clean_text(words_list):
    new_words = []
    for word in words_list:
        if word in en_stopwords:
            continue
        new_words.append(word)
    return new_words


def plot_features(features):
    num_examples = [(f_id, len(f_data)) for f_id, f_data in features.label2data.items()]
    num_examples.sort(key=lambda x: x[1], reverse=True)
    f_ids = [str(e[0]) for e in num_examples]
    f_counts = [e[1] for e in num_examples]

    num_large_features = len(np.where(np.array(f_counts) > 1000)[0])
    print('n_data > 1000', num_large_features)

    plt.bar(f_ids, f_counts)
    plt.show()


@dataclass
class Features:
    feature_type: str
    feature2data_weights: np.ndarray  # [num_feature, num_data]
    data2feature_weights: np.ndarray
    data2labels: dict
    label2data: dict
    num_features: int
    num_data: int

    @classmethod
    def load(cls, feature_file: str, feature_type: str):

        if feature_type == "topic":
            with open(feature_file, "r") as fp:
                saved = json.load(fp)

            feature2data_weights = np.array(saved["topic2doc_dist"])
            data2feature_weights = feature2data_weights.T

            num_features, num_data = feature2data_weights.shape
            labels = np.argmax(data2feature_weights, axis=-1)
            data2labels = {i: [labels[i]] for i in range(num_data)}
            label2data = {f: np.where(labels == f)[0] for f in range(num_features)}

            return cls(
                feature_type=feature_type,
                feature2data_weights=feature2data_weights,
                data2feature_weights=data2feature_weights,
                data2labels=data2labels,
                label2data=label2data,
                num_features=num_features,
                num_data=num_data,
            )
        else:
            all_data = []
            with open(feature_file, 'r') as fp:
                for line in fp.readlines():
                    data = json.loads(line.strip())
                    all_data.append(data)
            num_features, num_data = len(all_data[0]['theta']), len(all_data)
            data2feature_weights = np.array([data['theta'] for data in all_data])  # [num_data, num_features]
            feature2data_weights = data2feature_weights.T  # [num features, num_datat]
            data2labels = {data_id: np.where(np.array(all_data[data_id]['theta']) > 0)[0].tolist() for data_id in range(len(all_data))}
            label2data = {feature_id: [] for feature_id in range(num_features)}
            for data_id in data2labels:
                for label in data2labels[data_id]:
                    label2data[label].append(data_id)
            label2data = {k: np.array(v) for k, v in label2data.items()}

            return cls(
                feature_type=feature_type,
                feature2data_weights=feature2data_weights,
                data2feature_weights=data2feature_weights,
                data2labels=data2labels,
                label2data=label2data,
                num_features=num_features,
                num_data=num_data,
            )


class WikitextDataset(Dataset):
    def __init__(self, data_file):
        
        super(WikitextDataset, self).__init__()
        
        self.vocab = {}

        self.ids2pos = {}

        self.ids = []
        self.texts = []
        self.super_categories = []
        self.sub_categories = []
        self.categories = []
        self.page_names = []
        self.tokenized_texts = []

        with open(data_file, 'r') as fp:
            for i, line in enumerate(fp.readlines()):
                data = json.loads(line.strip())

                self.ids2pos[data['id']] = len(self.ids)

                self.ids.append(i)  # line number as id
                self.texts.append(data['text'])
                self.sub_categories.append(data['subcategory'])
                self.categories.append(data['category'])
                self.super_categories.append(data['supercategory'])
                self.page_names.append(data['page_name'])
                self.tokenized_texts.append(data['tokenized_text'].split(' '))

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'text': self.texts[idx],
            'tokenized_text': self.tokenized_texts[idx],
            'label': self.sub_categories[idx]
        }

    def get_by_id(self, doc_id):
        return self.__getitem__(self.ids2pos[doc_id])

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class BillDataset:
    def __init__(self, data_file):
        self.ids = []
        self.texts = []
        # self.super_categories = []
        self.sub_categories = []
        self.categories = []
        self.tokenized_texts = []

        self.vocab = set([])
        with open(data_file, 'r') as fp:
            for lid, line in enumerate(fp.readlines()):
                data = json.loads(line.strip())
                self.ids.append(lid)
                self.texts.append(data['summary'])
                self.categories.append(data['topic'])
                self.sub_categories.append(data['subtopic'])
                self.tokenized_texts.append(data['tokenized_text'].split(' '))
                self.vocab.update(self.tokenized_texts[-1])

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'text': self.texts[idx],
            'label': self.categories[idx],
            'sub_categories': self.sub_categories[idx],
            'tokenized_text': self.tokenized_texts[idx]
        }

    def __len__(self):
        return len(self.ids)


class SAEWikitext(Dataset):
    def __init__(self, data_file):
        super(SAEWikitext, self).__init__()

        self.data = []
        self.ids = []
        self.texts = []
        with open(data_file, 'r') as fp:
            for idx, line in enumerate(fp.readlines()):
                data = json.loads(line.strip())
                self.data.append(data)
                self.texts.append(data['text'])
                self.ids.append(idx)

        # print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    # sae_dataset = SAEDataset(data_file='data/SAE/sae_wikitext_train.jsonl')
    sae_features = Features.load(feature_file='output/topic_models/sae_wikitext_256.json', feature_type='sae')
    # tm_features = Features.load(feature_file='output/topic_models/lda_wikitext_25.json', feature_type='topic')
    plot_features(sae_features)
    pass
