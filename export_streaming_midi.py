import torch.nn as nn
import nn_tilde
import torch

from diffusion.model import RectifiedFlow
import argparse
import os

torch.set_grad_enabled(False)

import gin

import cached_conv as cc

cc.use_cached_conv(False)

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default="test_distill_midi")
parser.add_argument("--step", type=int, default=800000)
parser.add_argument("--out_name", type=str, default="AFTER_slakh_midi_acausal")
parser.add_argument("--emb_model_path",
                    type=str,
                    default="pretrained/slakh_2048_acausal.ts")
parser.add_argument("--npoly", type=int, default=4)


def main(args):

    # Parse model folder
    folder = "./runs/" + args.name
    checkpoint_path = folder + "/checkpoint" + str(args.step) + "_EMA.pt"
    config = folder + "/config.gin"

    out_name = args.out_name if args.out_name is not None else args.name

    # Parse config
    gin.parse_config_file(config)
    SR = gin.query_parameter("%SR")

    # Emb model

    # Instantiate model
    blender = RectifiedFlow()

    # Load checkpoints
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state"]
    blender.load_state_dict(state_dict, strict=False)

    # Emb model
    emb_model = torch.jit.load(args.emb_model_path).eval()
    blender.emb_model = emb_model

    # Send to device
    blender = blender.eval()

    # Get some parameters
    n_signal = gin.query_parameter('%N_SIGNAL_STREAMING')
    n_signal_timbre = gin.query_parameter('%N_SIGNAL')
    zt_channels = gin.query_parameter("%ZT_CHANNELS_POST")
    zs_channels = gin.query_parameter("%TOTAL_ZS_CHANNELS")
    ae_latents = gin.query_parameter("%IN_SIZE")

    # Trace the unet
    x = torch.ones(1, ae_latents, n_signal)
    time_cond = torch.randn(1, zs_channels, n_signal)
    cond = torch.randn(1, zt_channels)
    t = torch.ones(1)
    net = torch.jit.trace(blender.net, (x, t, time_cond, cond))

    class Streamer(nn_tilde.Module):

        def __init__(self) -> None:
            super().__init__()

            self.net = net
            self.encoder = blender.encoder
            self.post_encoder = blender.post_encoder
            self.encoder_time = blender.encoder_time

            self.npoly = args.npoly
            self.n_signal = n_signal
            self.n_signal_timbre = n_signal_timbre
            self.zs_channels = zs_channels
            self.zt_channels = zt_channels
            self.ae_latents = ae_latents
            self.emb_model_out = torch.jit.load(args.emb_model_path)
            #self.emb_model_structure = torch.jit.load(args.emb_model_path)
            self.emb_model_timbre = torch.jit.load(
                "pretrained/slakh_stream.ts")

            self.drop_value = blender.drop_value

            dummy = torch.zeros(1, 1, 4 * 4096)
            z = self.emb_model_out.encode(dummy)
            self.ae_ratio = 4 * 4096 // z.shape[-1]

            self.sr = gin.query_parameter("%SR")
            self.zt_buffer = self.n_signal_timbre * self.ae_ratio

            self.register_attribute("nb_steps", 1)
            self.register_attribute("learn_zsem", True)
            self.register_attribute("guidance", 1.)
            self.register_attribute("timbre_factor", 0.)

            self.register_buffer(
                "previous_chunks",
                self.drop_value *
                torch.ones(1, self.ae_latents, self.n_signal))

            self.register_buffer(
                "previous_timbre",
                torch.zeros(1, self.ae_latents,
                            self.n_signal_timbre - self.n_signal))

            self.register_buffer("last_zsem", torch.zeros(1, self.zt_channels))

            self.register_method(
                "forward",
                in_channels=1 + 2 * self.npoly,
                in_ratio=1,
                out_channels=1,
                out_ratio=1,
                input_labels=[
                    f"(signal) Input note_{i}" for i in range(2 * self.npoly)
                ] + [f"(signal) Input timbre"],
                output_labels=[f"(signal) Output"],
                test_buffer_size=self.n_signal * self.ae_ratio,
            )

            self.register_method(
                "forward_manual",
                in_channels=2 * self.npoly + self.zt_channels,
                in_ratio=1,
                out_channels=1,
                out_ratio=1,
                input_labels=[
                    f"(signal) Input note_{i}" for i in range(2 * self.npoly)
                ] + [
                    f"(signal_{i}) Input timbre"
                    for i in range(self.zt_channels)
                ],
                output_labels=[f"(signal) Output"],
                test_buffer_size=self.n_signal * self.ae_ratio,
            )

        @torch.jit.export
        def get_learn_zsem(self) -> bool:
            return self.learn_zsem[0]

        @torch.jit.export
        def set_learn_zsem(self, learn_zsem: bool) -> int:
            self.learn_zsem = (learn_zsem, )
            return 0

        @torch.jit.export
        def get_guidance(self) -> float:
            return self.guidance[0]

        @torch.jit.export
        def set_guidance(self, guidance: float) -> int:
            self.guidance = (guidance, )
            return 0

        @torch.jit.export
        def get_timbre_factor(self) -> float:
            return self.timbre_factor[0]

        @torch.jit.export
        def set_timbre_factor(self, guidance: float) -> int:
            self.timbre_factor = (guidance, )
            return 0

        @torch.jit.export
        def get_nb_steps(self) -> int:
            return self.nb_steps[0]

        @torch.jit.export
        def set_nb_steps(self, nb_steps: int) -> int:
            self.nb_steps = (nb_steps, )
            return 0

        def model_forward(self, x: torch.Tensor, time: torch.Tensor,
                          cond: torch.Tensor, time_cond: torch.Tensor,
                          time_cond_add: torch.Tensor,
                          guidance_cond_factor: float,
                          guidance_joint_factor: float, total_guidance: float):

            full_time = time.repeat(4, 1, 1)
            full_x = x.repeat(4, 1, 1)
            full_cond = torch.cat([
                cond, self.drop_value * torch.ones_like(cond),
                self.drop_value * torch.ones_like(cond), cond
            ])
            full_time_cond = torch.cat([
                time_cond, self.drop_value * torch.ones_like(time_cond),
                time_cond, self.drop_value * torch.ones_like(time_cond)
            ])

            if time_cond_add is not None:
                time_cond_add = time_cond_add.repeat(4, 1, 1)
                full_time_cond = torch.cat([full_time_cond, time_cond_add],
                                           dim=1)

            dx = self.net(full_x,
                          time=full_time.view(-1),
                          cond=full_cond,
                          time_cond=full_time_cond)

            dx_full, dx_none, dx_time_cond, dx_cond = torch.chunk(dx, 4, dim=0)

            dx = dx_none + total_guidance * (guidance_joint_factor *
                                             (dx_full - dx_none) +
                                             (1 - guidance_joint_factor) *
                                             (guidance_cond_factor *
                                              (dx_cond - dx_none) +
                                              (1 - guidance_cond_factor) *
                                              (dx_time_cond - dx_none)))
            return dx

        def sample(self, x0: torch.Tensor, cond: torch.Tensor,
                   time_cond: torch.Tensor, time_cond_add: torch.Tensor,
                   nb_steps: int, total_guidance: float, timbre_factor: float):

            dt = 1 / nb_steps
            t_values = torch.linspace(0, 1, nb_steps + 1)[:-1]
            x = x0

            guidance_joint_factor = 1. - 0.7 * abs(timbre_factor)
            guidance_cond_factor = 0.5 + 0.5 * timbre_factor

            for t in t_values:
                t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
                x = x + self.model_forward(
                    x=x,
                    time=t,
                    cond=cond,
                    time_cond=time_cond,
                    guidance_cond_factor=guidance_cond_factor,
                    guidance_joint_factor=guidance_joint_factor,
                    total_guidance=total_guidance,
                    time_cond_add=time_cond_add) * dt
            return x

        @torch.jit.export
        def encode(self, x) -> torch.Tensor:
            x = self.emb_model_timbre.encode(x)
            append_length = x.shape[-1]

            x = torch.cat((self.previous_timbre, x), -1)
            self.previous_timbre = x[..., append_length:]

            zsem = self.encoder(self.previous_timbre)
            zsem = self.post_encoder(
                zsem) if self.post_encoder is not None else zsem
            self.last_zsem = zsem

            return zsem

        @torch.jit.export
        def encode_time(self, x) -> torch.Tensor:
            #x = self.emb_model_structure.encode(x)
            x = self.encoder_time(x)
            return x

        @torch.jit.export
        def forward_manual(self, x: torch.Tensor) -> torch.Tensor:

            n = x.shape[0]

            if self.previous_chunks.shape[
                    0] != n or self.previous_timbre.shape[0] != n:
                self.previous_chunks = self.previous_chunks.repeat(n, 1, 1)[:n]

            notes = x[:, :2 * self.npoly]
            if notes.shape[-1] != self.n_signal:
                notes = torch.nn.functional.interpolate(notes,
                                                        size=self.n_signal,
                                                        mode="nearest")
            time_cond = torch.zeros((n, 128, self.n_signal))

            for i in range(self.npoly):
                for j in range(self.n_signal):
                    if notes[0, 2 * i + 1, j] > 0:
                        time_cond[:, notes[:, 2 * i].long(),
                                  j] = notes[:, 2 * i + 1, j] / 128

            time_cond = self.encode_time(time_cond)

            zsem = x[:, 2 * self.npoly:].mean(-1)

            x0 = torch.randn(n, self.ae_latents, time_cond.shape[-1])

            xS = self.sample(x0,
                             time_cond=time_cond,
                             time_cond_add=self.previous_chunks,
                             cond=zsem,
                             nb_steps=self.nb_steps[0],
                             total_guidance=self.guidance[0],
                             timbre_factor=self.timbre_factor[0])

            self.previous_chunks = xS.clone()
            xS = self.emb_model_out.decode(xS)
            return xS

        @torch.jit.export
        def forward(self, x: torch.Tensor) -> torch.Tensor:

            n = x.shape[0]

            if self.previous_chunks.shape[
                    0] != n or self.previous_timbre.shape[0] != n:
                self.previous_chunks = self.previous_chunks.repeat(n, 1, 1)[:n]
                self.last_zsem = self.last_zsem.repeat(n, 1)[:n]
                self.previous_timbre = self.previous_timbre.repeat(n, 1, 1)[:n]

            notes = x[:, :2 * self.npoly]
            if notes.shape[-1] != self.n_signal:
                notes = torch.nn.functional.interpolate(notes,
                                                        size=self.n_signal,
                                                        mode="nearest")
            time_cond = torch.zeros((n, 128, self.n_signal))

            for i in range(self.npoly):
                for j in range(self.n_signal):
                    if notes[0, 2 * i + 1, j] > 0:
                        time_cond[:, notes[:, 2 * i].long(),
                                  j] = notes[:, 2 * i + 1, j] / 128

            time_cond = self.encode_time(time_cond)

            if self.learn_zsem[0] == True:
                zsem = self.encode(x[:, -1:])
                self.last_zsem = zsem
            else:
                zsem = self.last_zsem
                print("using saved zsem")

            x0 = torch.randn(n, self.ae_latents, time_cond.shape[-1])

            xS = self.sample(x0,
                             time_cond=time_cond,
                             time_cond_add=self.previous_chunks,
                             cond=zsem,
                             nb_steps=self.nb_steps[0],
                             total_guidance=self.guidance[0],
                             timbre_factor=self.timbre_factor[0])

            self.previous_chunks = xS.clone()
            xS = self.emb_model_out.decode(xS)
            return xS

    ###

    ####
    streamer = Streamer()
    os.makedirs("./exports/", exist_ok=True)
    streamer.export_to_ts("./exports/" + out_name + ".ts")

    print("SUUUUUUCEEEEESSSSSSS")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
