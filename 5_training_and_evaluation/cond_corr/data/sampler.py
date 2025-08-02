import torch
import random

from torch.utils.data import Sampler

class BatchSampler(Sampler):
    def __init__(self, label_dict, n_iter, min_img, n_sets, seed=10):

        self.n_iter = n_iter
        self.min_img = min_img
        self.n_sets = n_sets
        self.label_dict = {k:torch.where(v)[0].tolist() for k,v in label_dict.items()}

        self.aff_categs = list(label_dict.keys())
        self.rng = random.Random(seed)
        self.default_state = self.rng.getstate()

    def reset_rng(self):
        self.rng.setstate(self.default_state)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch = []
            batch_categs = self.rng.choices(self.aff_categs, k=self.n_sets) # with replacement
            # batch_categs = random.sample(self.aff_categs, k=self.n_sets)

            for categ in batch_categs:
                categ_indices = self.label_dict[categ]
                batch.extend(self.rng.sample(categ_indices, k=self.min_img)) # without replacement

            yield iter(batch)


class SpatialBatchSampler(Sampler):
    def __init__(self, num_pairs, n_iter, n_sets, seed=10):

        self.n_iter = n_iter
        self.n_sets = n_sets
        self.num_pairs = num_pairs
        self.rng = random.Random(seed)
        self.default_state = self.rng.getstate()

    def reset_rng(self):
        self.rng.setstate(self.default_state)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch = []

            batch_indices = self.rng.choices(range(self.num_pairs), k=self.n_sets) # with replacement
            for index in batch_indices:
                batch.extend([index])

            yield iter(batch)