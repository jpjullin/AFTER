import torch
import gin
import numpy as np


def crop(arrays, length, idxs):
    return [
        torch.stack([xc[..., i:i + length] for i, xc in zip(idxs, array)])
        for array in arrays
    ]


def normalize(array):
    return (array - array.min()) / (array.max() - array.min() + 1e-6)


@gin.configurable
def collate_fn(batch, n_signal, structure_type, ae_ratio):
    x = torch.from_numpy(np.stack([b["z"] for b in batch], axis=0))

    i0 = np.random.randint(0, x.shape[-1] - n_signal, x.shape[0])
    x_target = crop([x], n_signal, i0)[0]

    i1 = np.random.randint(0, x.shape[-1] - n_signal, x.shape[0])
    x_timbre = crop([x], n_signal, i1)[0]

    #if blender.time_transform is not None:
    #    x_waveform = torch.from_numpy(
    #        np.stack([b["waveform"] for b in batch], axis=0))
    #    x_waveform = x_waveform.reshape(x_waveform.shape[0], 1, -1)
    #    time_cond_target = crop([x_waveform], n_signal * ae_ratio,
    #                            i0 * ae_ratio)[0]
    if structure_type == "audio":
        time_cond_target = x_target
    elif structure_type == "midi":
        midi = [b["midi"] for b in batch]

        times = np.linspace(
            0, x.shape[-1] * ae_ratio / gin.query_parameter("%SR"),
            x.shape[-1])
        pr = [m.get_piano_roll(times=times) for m in midi]

        pr = map(normalize, pr)
        pr = np.stack(list(pr))
        pr = torch.from_numpy(pr).float()
        pr = torch.stack([prc[..., i:i + n_signal] for i, prc in zip(i0, pr)])
        time_cond_target = pr

    return {"x": x_target, "x_cond": x_timbre, "x_time_cond": time_cond_target}


@gin.configurable
def collate_fn_streaming(batch, n_signal, n_signal_timbre, structure_type,
                         ae_ratio):
    margin = n_signal_timbre - n_signal
    x = torch.from_numpy(np.stack([b["z"] for b in batch], axis=0))
    i0 = np.random.randint(margin, x.shape[-1] - n_signal, x.shape[0])
    x_target = crop([x], n_signal, i0)[0]

    x_past = crop([x], n_signal, i0 - n_signal)[0]
    i1 = np.random.randint(0, x.shape[-1] - n_signal_timbre, x.shape[0])
    x_timbre = crop([x], n_signal_timbre, i1)[0]

    if structure_type == "audio":
        x_target_margin = crop([x], n_signal + margin, i0 - margin)[0]
        time_cond_target = x_target_margin

    elif structure_type == "midi":
        midi = [b["midi"] for b in batch]

        times = np.linspace(
            0, x.shape[-1] * ae_ratio / gin.query_parameter("%SR"),
            x.shape[-1])
        pr = [m.get_piano_roll(times=times) for m in midi]

        pr = map(normalize, pr)
        pr = np.stack(list(pr))
        pr = torch.from_numpy(pr).float()
        pr = torch.stack(
            [prc[..., i - margin:i + n_signal] for i, prc in zip(i0, pr)])
        time_cond_target = pr

    return {
        "x": x_target,
        "x_cond": x_timbre,
        "x_time_cond": time_cond_target,
        "x_time_cond_additional": x_past,
        "margin": margin,
    }
