import json
import sys
import nltk
from collections import OrderedDict
from nltk.corpus import stopwords
from torch.utils.data import Dataset
en_stopwords = set(stopwords.words('english'))


def clean_text(words_list):
    new_words = []
    for word in words_list:
        if word in en_stopwords:
            continue
        new_words.append(word)
    return new_words


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

                self.ids.append(i)
                self.texts.append(data['text'])
                self.sub_categories.append(data['subcategory'])
                self.page_names.append(data['page_name'])
                # self.tokenized_texts.append(clean_text(nltk.wordpunct_tokenize(data['text'].lower())))
                self.tokenized_texts.append(data['tokenized_text'].split(' '))

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'text': self.texts[idx],
            'tokenized_text': self.tokenized_texts[idx]
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
        pass


    def __getitem__(self):
        return

class ClusterDataset:
    def __init__(self):
        self.clusters = OrderedDict()  # {cluster id: {document_id: score},  }


if __name__ == '__main__':
    pass
