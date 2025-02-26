

import gin
gin.add_config_file_search_path('./diffusion/configs')

import torch
import os
import numpy as np
from dataset import CachedSimpleDataset, CombinedDataset
import argparse
from diffusion.utils import collate_fn_streaming

parser = argparse.ArgumentParser()

# MODEL
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--restart", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument('--config', action="append", default=[])
parser.add_argument('--model', default="rectified")

# Model to distill
parser.add_argument("--pretrained_model", type=str)
parser.add_argument("--pretrained_step", type=str)

# Training
parser.add_argument("--bsize", type=int, default=256)

# DATASET
parser.add_argument(
    "--db_path",
    type=str,
    default=None
)

parser.add_argument("--db_folder", type=str, default=None)

parser.add_argument("--out_path", type=str, default="./diffusion/runs")
parser.add_argument("--emb_model_path",
                    type=str,
                    default=None)

parser.add_argument("--use_cache",
                    type=bool,
                    default=False,
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--recache_every", type=int, default=None)
parser.add_argument("--max_samples", type=int, default=None)

parser.add_argument("--num_workers", type=int, default=8)

def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def main(args):

    encoders_config_path = "./diffusion/runs/" + args.pretrained_model + "/config.gin"

    with gin.unlock_config():
        gin.parse_config_files_and_bindings([encoders_config_path], [])

    gin.bind_parameter(
        "%TOTAL_ZS_CHANNELS",
        gin.query_parameter("%ZS_CHANNELS") + gin.query_parameter("%IN_SIZE"))

    gin.parse_config_files_and_bindings(
        map(add_gin_extension, args.config),
        [],
    )
    
    if args.restart > 0:
        config_path = "./diffusion/runs/" + args.name + "/config.gin"

        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [])
    

    device = "cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu"

    ######### BUILD MODEL #########

    if args.emb_model_path == "music2latent":
        from music2latent import EncoderDecoder
        emb_model = EncoderDecoder(device=device)
        ae_ratio = 4096
    else:
        emb_model = torch.jit.load(args.emb_model_path)#.to(device)
        dummy = torch.randn(1, 1, 4096)#.to(device)
        z = emb_model.encode(dummy)
        ae_ratio = 4096 // z.shape[-1]
        with gin.unlock_config():
            gin.bind_parameter("diffusion.utils.collate_fn_streaming.ae_ratio", ae_ratio)

    if args.model == "rectified":
        from diffusion.model import RectifiedFlow
        blender = RectifiedFlow(device=device, emb_model=emb_model)
    elif args.model == "edm":
        from diffusion.model import EDM
        blender = EDM(device=device, emb_model=emb_model)
    elif args.model == "sCM":
        from diffusion.model import sCM
        blender = sCM(device=device, emb_model=emb_model)
    else:
        raise ValueError("Model not recognized")
    
    ######### LOAD THE CHECKPOINT #########

    state_dict = torch.load(f"./diffusion/runs/{args.pretrained_model}/checkpoint{args.pretrained_step}_EMA.pt",
                            map_location="cpu")

    state_dict_model = {
        key: value
        for key, value in state_dict["model_state"].items()
        if "encoder." in key or "encoder_time." in key
    }

    blender.load_state_dict(state_dict_model, strict=False)
    
    blender.encoder.eval()
    blender.encoder_time.eval()

    ######### GET THE DATASET #########
    
    structure_type = gin.query_parameter("%STRUCTURE_TYPE")
    data_keys = ["z"
                 ] + (["waveform"] if blender.time_transform is not None else
                      []) + (["midi"] if structure_type == "midi" else [])
                 
                 

    if args.db_folder is not None:
        main_folder = args.db_folder
        audio_folders = [
            os.path.join(main_folder, f) for f in os.listdir(main_folder)
        ]
        db_paths = [f + "/ae_44k" for f in audio_folders]

        path_dict = {f: {"name": f, "path": f} for f in db_paths}

        dataset = CombinedDataset(
            path_dict=path_dict,
            freqs="estimate",
            config="train",
            keys= data_keys,
        )

        train_sampler = dataset.get_sampler()
        valset = CombinedDataset(path_dict=path_dict, config="val", keys= data_keys)
        val_sampler = valset.get_sampler()

    else:
        dataset = CachedSimpleDataset(path=args.db_path,
                                      keys=data_keys,
                                      max_samples=args.max_samples,
                                      recache_every=args.recache_every,
                                      init_cache=args.use_cache,
                                      split = "train")

        valset = CachedSimpleDataset(path=args.db_path,
                                     keys=data_keys,
                                     max_samples=args.max_samples,
                                     recache_every=args.recache_every,
                                     split="validation",
                                     init_cache=args.use_cache)
        train_sampler, val_sampler = None, None


    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bsize,
        shuffle=True if train_sampler is None else False,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn_streaming,
        sampler=train_sampler if train_sampler is not None else None)
    
    valid_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.bsize,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_fn_streaming,
        sampler=val_sampler if val_sampler is not None else None)
    
    print(next(iter(train_loader))["x"].shape)
    
    
    # Quick trick to make sure the gin parameters are correctly saved to the operative config
    try:
        dummy = collate_fn_streaming([])
    except:
        pass
    
    ######### SAVE DIR #########
    model_dir = os.path.join(args.out_path, args.name)
    os.makedirs(model_dir, exist_ok=True)

    ######### PRINT NUMBER OF PARAMETERS #########
    num_el = 0
    for p in blender.net.parameters():
        num_el += p.numel()
    print("Number of parameters - unet : ", num_el / 1e6, "M")

    if blender.encoder is not None:
        num_el = 0
        for p in blender.encoder.parameters():
            num_el += p.numel()
        print("Number of parameters - encoder : ", num_el / 1e6, "M")

    if blender.encoder_time is not None:
        num_el = 0
        for p in blender.encoder_time.parameters():
            num_el += p.numel()
        print("Number of parameters - encoder_time : ", num_el / 1e6, "M")

    if blender.classifier is not None:
        num_el = 0
        for p in blender.classifier.parameters():
            num_el += p.numel()
        print("Number of parameters - classifier : ", num_el / 1e6, "M")

    ######### TRAINING #########
    d = {
        "model_dir": model_dir,
        "dataloader": train_loader,
        "validloader": valid_loader,
        "restart_step": args.restart,
    }

    blender.fit(**d)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
