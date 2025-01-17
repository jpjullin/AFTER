import torch
import numpy as np
from dataset.transforms import random_phase_mangle
from random import random
import cached_conv as cc
import gin
import os

gin.add_config_file_search_path('./autoencoder/configs')

from autoencoder.networks.SimpleNetsStream import AutoEncoder
from autoencoder.trainer import Trainer
from dataset.dataset import CachedSimpleDataset, CombinedDataset

cc.use_cached_conv(False)
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default="test")

parser.add_argument("--db_path",
                    type=str,
                    default="/data/nils/datasets/slakh/train")
parser.add_argument("--db_folder", type=str, default=None)

parser.add_argument('--config', action="append", default=[])
parser.add_argument("--restart", type=int, default=0)

parser.add_argument("--bsize", type=int, default=6)
parser.add_argument("--num_signal", type=int, default=131072)
parser.add_argument("--gpu", type=int, default=-1)

parser.add_argument("--use_cache",
                    type=bool,
                    default=False,
                    action=argparse.BooleanOptionalAction)


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def main(args):
    model_name = args.name
    num_signal = args.num_signal
    step_restart = args.restart

    device = "cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu"

    ## GIN CONFIG
    if args.restart > 0:
        config_path = "./runs/" + model_name + "/config.gin"
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [])

    else:
        gin.parse_config_files_and_bindings(
            map(add_gin_extension, args.config),
            [],
        )

    sr = gin.query_parameter("%SR")
    #MODELS
    autoencoder = AutoEncoder()

    ### TEST NETWORKS
    x = torch.randn(1, 1, 4096 * 16)
    z = autoencoder.encode(x)
    print(z.shape)
    y = autoencoder.decode(z)

    assert x.shape == y.shape, ValueError(
        f"Shape mismatch: x.shape = {x.shape}, y.shape = {y.shape}")

    ## Start the training
    trainer = Trainer(autoencoder, device=device)

    num_el = 0
    for p in autoencoder.encoder.parameters():
        num_el += p.numel()
    print("Number of parameters - Encoder : ", num_el / 1e6, "M")

    num_el = 0
    for p in autoencoder.decoder.parameters():
        num_el += p.numel()
    print("Number of parameters - Decoder : ", num_el / 1e6, "M")

    num_el = 0
    for p in trainer.discriminator.parameters():
        num_el += p.numel()
    print("Number of parameters - Discriminator : ", num_el / 1e6, "M")

    ## DATASET

    transforms = []

    class RandomApply():
        """
        Apply transform with probability p
        """

        def __init__(self, transform, p=.5):
            self.transform = transform
            self.p = p

        def __call__(self, x: np.ndarray):
            if random() < self.p:
                x = self.transform(x)
            return x

    class RandomGain():

        def __init__(self, db):
            """
            Args:
                db: randomize gain from -db to db. upper bound will be clipped
                    to prevent peak > 1.
            """
            self.db = db

        def __call__(self, x: np.ndarray):
            gain = 10**((random() * (-self.db)) / 20)
            return x * gain

    transforms = [
        RandomApply(lambda x: random_phase_mangle(x, 20, 2000, .99, sr),
                    p=0.8),
        RandomApply(RandomGain(20), p=0.8)
    ]

    def collate_fn(x):
        x = [l["waveform"] for l in x]
        x = [
            l[..., i0:i0 + num_signal] for l, i0 in zip(
                x, torch.randint(x[0].shape[-1] - num_signal, (len(x), )))
        ]

        x = np.stack(x)
        for transform in transforms:
            x = transform(x)
        x = torch.from_numpy(x).reshape(x.shape[0], 1, -1).float()
        return x

    if args.db_folder is not None:
        main_folder = args.db_folder
        audio_folders = [
            os.path.join(main_folder, f) for f in os.listdir(main_folder)
        ]
        db_paths = [f + "/ae_44k" for f in audio_folders]

        path_dict = {f: {"name": f, "path": f} for f in db_paths}

        dataset = CombinedDataset(
            path_dict=path_dict,
            keys=["waveform"],
            freqs="estimate",
            config="train",
            init_cache=args.use_cache,
        )

        train_sampler = dataset.get_sampler()
        valset = CombinedDataset(
            path_dict=path_dict,
            config="val",
            freqs="estimate",
            keys=["waveform"],
            init_cache=args.use_cache,
        )
        val_sampler = valset.get_sampler()

    else:
        dataset = CachedSimpleDataset(path=args.db_path,
                                      keys=["waveform"],
                                      init_cache=args.use_cache)

        valset = CachedSimpleDataset(path=args.db_path,
                                     keys=["waveform"],
                                     validation=True,
                                     init_cache=args.use_cache)

        train_sampler, val_sampler = None, None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bsize,
        shuffle=True if train_sampler is None else False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0,
        sampler=train_sampler)

    validloader = torch.utils.data.DataLoader(valset,
                                              batch_size=args.bsize,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              drop_last=True,
                                              num_workers=0,
                                              sampler=val_sampler)

    x = next(iter(dataloader))
    print("Training size : ", x.shape)

    if step_restart > 0:
        print("Loading model from step ", step_restart)
        path = "./runs/" + model_name
        trainer.load_model(path, step_restart, args.restart_load_discrim)

    trainer.fit(dataloader,
                validloader,
                tensorboard="./autoencoder/runs/" + model_name)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
