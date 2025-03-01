import torch
import lmdb
from .audio_example import AudioExample
from random import random
from tqdm import tqdm
import numpy as np


class CachedSimpleDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        path,
        keys=['waveform', 'metadata'],
        max_samples=None,
        num_sequential=100,
        recache_every=None,
        init_cache=False,
        validation_size=0.1,
        split=None,
        readonly = True,
    ) -> None:
        super().__init__()


        self.num_sequential = num_sequential
        self.buffer_keys = keys
        self.max_samples = max_samples
        self.recache_every = recache_every
        self.recache_counter = 0

        self.env = lmdb.open(path,
                             lock=False,
                             readonly=readonly,
                             readahead=True,
                             map_async=False,
                             map_size=200*1024**3 if readonly == False else None)

        with self.env.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

        train_ids, valid_ids = train_test_split(list(range(len(self.keys))),
                                                test_size=validation_size,
                                                random_state=42)

        if split == "validation":
            self.keys = [self.keys[i] for i in valid_ids]
        elif split == "train":
            self.keys = [self.keys[i] for i in train_ids]
        

        if self.max_samples is not None and self.max_samples < len(self.keys):
            np.random.seed(0)
            self.keys = np.random.choice(self.keys,
                                         self.max_samples,
                                         replace=False)
        else:
            self.max_samples = None

        self.indexes = list(range(len(self.keys)))
        self.cached = False

        if init_cache:
            self.build_cache()

    def __len__(self):
        return len(self.indexes)

    def build_cache(self):
        self.cached = False
        self.cache = []

        self.indexes = list(range(len(self.keys)))
        if self.max_samples is not None:

            self.indexes_start = np.random.choice(
                self.indexes[:-self.num_sequential],
                self.max_samples // self.num_sequential,
                replace=False)

            self.indexes = [
                start + i for start in self.indexes_start
                for i in range(self.num_sequential)
            ]

        for i in tqdm(range(len(self.indexes))):
            self.cache.append(self.__getitem__(i))

        self.cached = True

    def __getitem__(self, index):
        if self.cached == True:
            self.recache_counter += 1
            if self.recache_every is not None and self.recache_counter == self.recache_every:
                self.build_cache()
                self.recache_counter = 0

            return self.cache[index]

        index = self.indexes[index]

        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[index]))

        out = {}
        for key in self.buffer_keys:
            if key == "metadata":
                out[key] = ae.get_metadata()
            else:
                try:
                    out[key] = ae.get(key)
                except:
                    print("key: ", key, " not found")

        if "midi" in out.keys():
            midi = out["midi"]

            out["midi"] = midi

        return out


from sklearn.model_selection import train_test_split


class CombinedDataset(torch.utils.data.Dataset):

    def __init__(self,
                 path_dict=None,
                 dataset_dict=None,
                 keys=["waveform"],
                 transforms=[],
                 config="all",
                 num_samples=None,
                 freqs=None,
                 init_cache=False,
                 **kwargs):
        super().__init__()
        self.config = config

        if dataset_dict is not None:
            self.datasets = {k: v["dataset"] for k, v in dataset_dict.items()}
            info_dict = dataset_dict

        elif path_dict is not None:
            self.datasets = {
                k:
                CachedSimpleDataset(
                    v["path"],
                    keys=keys,
                    max_samples=num_samples,
                    init_cache=init_cache,
                    validation=True if config == "val" else False)
                for k, v in path_dict.items()
            }
            info_dict = path_dict
        else:
            print("provide either path or dataset dict")

        if freqs == "estimate":
            for k in info_dict.keys():
                info_dict[k]["freq"] = len(self.datasets[k])**0.3

        for k, v in self.datasets.items():
            print(k, " : ", len(v))

        #if config == "train":
        self.keys = {k: v.keys for k, v in self.datasets.items()}

        #
        #elif config == "val":
        #    self.keys = {
        #        k: train_test_split(v.keys, test_size=0.05, random_state=42)[1]
        #        for k, v in self.datasets.items()
        #    }
        #
        #else:
        #    self.keys = {k: v.keys for k, v in self.datasets.items()}

        self.len = int(np.sum([len(v) for v in self.keys.values()]))

        self.weights = {
            k: info_dict[k]["freq"] * self.len / len(v)
            for k, v in self.keys.items()
        }

        self.dataset_ids = []
        self.weights_indexes = []
        self.all_keys = []
        self.all_indexes = []

        self.transforms = transforms

        for i, k in enumerate(self.keys.keys()):
            self.dataset_ids = self.dataset_ids + [k] * len(self.keys[k])
            self.weights_indexes += [self.weights[k]] * len(self.keys[k])
            self.all_keys.extend(self.keys[k])
            self.all_indexes.extend(list(range(len(self.keys[k]))))
        self.cache = False

    def __len__(self):
        return int(self.len)

    def get_sampler(self):
        if self.config == "train":
            return torch.utils.data.WeightedRandomSampler(self.weights_indexes,
                                                          self.len,
                                                          replacement=True,
                                                          generator=None)
        elif self.config == "val":
            return torch.utils.data.WeightedRandomSampler(
                self.weights_indexes,
                self.len,
                replacement=True,
                generator=torch.Generator().manual_seed(42))

    def build_cache(self):
        print("building cache")
        self.data = {}
        for k in self.datasets.keys():
            datalist = []
            for idx in tqdm(range(len(self.keys[k]))):
                datalist.append(self.datasets[k][idx])
            self.data[k] = datalist

        self.cache = True

    def __getitem__(self, idx):

        dataset_id = self.dataset_ids[idx]
        data = self.datasets[dataset_id][self.all_indexes[idx]]
        data["label"] = dataset_id

        return data
