import nn_tilde
import torch
import cached_conv as cc
from autoencoder.networks.SimpleNetsStream import AutoEncoder

import argparse
import gin

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, default=3000000)
parser.add_argument("--name", type=str)
parser.add_argument("--save_name", type=str)
parser.add_argument("--stream",
                    type=bool,
                    default=False,
                    action=argparse.BooleanOptionalAction)


class AE(nn_tilde.Module):

    def __init__(self, model_name, step) -> None:
        super().__init__()
        config = "./autoencoder/runs/" + model_name + "/config.gin"

        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                [config],
                [],
            )

        sr = gin.query_parameter("%SR")
        print(sr)

        model = AutoEncoder()
        path = "./autoencoder/runs/" + model_name
        d = torch.load(path + "/checkpoint" + str(step) + ".pt")
        model.load_state_dict(d["model_state"])

        self.model = model

        test_array = torch.zeros((3, 1, 4096))
        z = self.model.encode(test_array)

        self.comp_ratio = test_array.shape[-1] // z.shape[-1]

        self.latent_size = gin.query_parameter("%LATENT_SIZE")
        self.target_channels = 1

        self.register_method(
            "encode",
            in_channels=self.target_channels,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=self.comp_ratio,
            input_labels=['(signal) input 1'],
            output_labels=[f"latent {i}" for i in range(self.latent_size)],
            test_buffer_size=self.comp_ratio,
        )

        self.register_method("decode",
                             in_channels=self.latent_size,
                             in_ratio=self.comp_ratio,
                             out_channels=self.target_channels,
                             out_ratio=1,
                             input_labels=[
                                 f'(signal) Latent dimension {i+1}'
                                 for i in range(self.latent_size)
                             ],
                             output_labels=[
                                 '(signal) Channel %d' % d
                                 for d in range(1, self.target_channels + 1)
                             ])

        self.register_method("forward",
                             in_channels=1,
                             in_ratio=1,
                             out_channels=1,
                             out_ratio=1,
                             input_labels=['(signal) input 1'],
                             output_labels=[
                                 '(signal) Channel %d' % d
                                 for d in range(1, self.target_channels + 1)
                             ])

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.use_pqmf:
            x = self.model.pqmf(x)

        z = self.model.encoder(x)

        z = self.model.bottleneck(z)

        x = self.model.decoder(z)

        if self.model.use_pqmf:
            x = self.model.pqmf.inverse(x)

        return x

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.use_pqmf:
            x = self.model.pqmf(x)
        z = self.model.encoder(x)
        z = self.model.bottleneck(z)

        return z

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.model.decoder(z)
        if self.model.use_pqmf:
            x = self.model.pqmf.inverse(x)
        return x


def main(argv):

    cc.use_cached_conv(argv.stream)

    ae = AE(model_name=argv.name, step=argv.step)

    test_array = torch.zeros((3, 1, ae.comp_ratio * 8))
    z = ae.encode(test_array)
    x = ae.decode(z)
    print(z.shape, x.shape)
    print(ae.forward(test_array).shape)

    ae.export_to_ts("./pretrained/" + argv.save_name +
                    ("_stream" if argv.stream else "") + ".ts")
    print("Success !")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
