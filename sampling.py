"""
Subset selection strategy
"""

import random
import numpy as np


class Selector:
    def __init__(self, dataset):
        self.dataset = dataset

    def select(self, sel_func, doc_weights, k):
        if sel_func == 'random':
            return self.random(k)
        elif sel_func == 'top_k':
            return self.top_k(k, doc_weights)

        return None

    def random(self, k):
        indices = random.sample([i for i in range(len(self.dataset))], k)
        return [self.dataset[i] for i in indices]

    def top_k(self, k, weights):
        sorted_indices = np.argsort(weights)
        return [self.dataset[sorted_indices[-i]] for i in range(1, k+1)]

    def stratified(self, k, weights):
        pass

    def utility(self, k, weights):
        pass
