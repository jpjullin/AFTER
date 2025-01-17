from typing import Callable, Optional
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gin
from torch_ema import ExponentialMovingAverage
import os

from einops import reduce, rearrange


@gin.configurable
class Base(nn.Module):

    def __init__(self,
                 net,
                 sr,
                 encoder=None,
                 post_encoder=None,
                 encoder_time=None,
                 classifier=None,
                 emb_model=None,
                 time_transform=None,
                 vector_quantizer=None,
                 drop_value=-4.,
                 drop_rate=0.2,
                 sigma_data="estimate",
                 device="cpu"):
        super().__init__()

        self.net = net
        self.encoder = encoder
        self.encoder_time = encoder_time
        self.post_encoder = post_encoder
        self.classifier = classifier

        self.vector_quantizer = vector_quantizer

        self.time_transform = time_transform
        self.sr = sr

        self.drop_value = drop_value
        self.drop_rate = drop_rate

        self.extra_modules = nn.ModuleDict({}).to(self.device)

        if sigma_data == "estimate":
            sigma_data = 0.
        self.register_buffer("sigma_data", torch.tensor(sigma_data))
        self.to(device)

        self.emb_model = emb_model

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(self,
               x0: torch.Tensor,
               nb_step: int,
               time_cond=None,
               cond=None,
               guidance: float = 1):
        pass

    def training_step(self, batch, step):
        pass

    def cfgdrop(self, datas, bsize, drop_targets=[], drop_rate=0.2):
        draw = torch.rand(bsize)
        test_drop_all = (draw < drop_rate)

        for i in range(len(datas)):
            test_drop_i = (draw > drop_rate * (i + 1)) & (draw < drop_rate *
                                                          (i + 2))

            test_drop = test_drop_all + test_drop_i if i in drop_targets else test_drop_all
            anti_test_drop = ~test_drop

            test_drop = self.broadcast_to(test_drop.to(datas[i]),
                                          datas[i].shape)
            anti_test_drop = self.broadcast_to(anti_test_drop.to(datas[i]),
                                               datas[i].shape)

            if datas[i] is None:
                datas[i] = None
            else:
                datas[i] = anti_test_drop * datas[
                    i] + test_drop * torch.ones_like(
                        datas[i]) * self.drop_value

        return datas

    def broadcast_to(self, alpha, shape):
        assert type(shape) == torch.Size
        return alpha.reshape(-1, *((1, ) * (len(shape) - 1)))

    def init_train(self, lr, dataloader, train_encoders):
        params = list(self.net.parameters())

        if self.encoder is not None and train_encoders[0]:
            print("trtaining encoder")
            params += list(self.encoder.parameters())

        if self.post_encoder is not None:
            print("trtaining encoder")
            params += list(self.post_encoder.parameters())

        if self.encoder_time is not None and train_encoders[1]:
            print("training encoder_time")
            params += list(self.encoder_time.parameters())

        if self.cycle_emb is not None:
            params += list(self.cycle_emb.parameters())

        if self.classifier is not None:
            self.opt_classifier = AdamW(self.classifier.parameters(),
                                        lr=lr,
                                        betas=(0.9, 0.999))

        for k, v in self.extra_modules.items():
            params += list(v.parameters())

        self.opt = AdamW(params, lr=lr, betas=(0.9, 0.999))
        self.step = 0
        self.ema_loss_averages = {}

        self.estimate_std(dataloader)

    @torch.no_grad()
    def estimate_std(self, loader, n_batches=64):
        xlist = []

        if self.sigma_data == torch.tensor(0.):
            print("Estimating sigma...")

            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                x = batch["x"]
                xlist.append(x)
            x = torch.cat(xlist, dim=0)
            sigma = x.std()
            print(f"setting sigma_data to {sigma}")
            self.sigma_data = torch.tensor(sigma.item())

    @gin.configurable
    @torch.no_grad()
    def prep_data(self, batch, device=None):

        x1 = batch["x"].to(device)
        x1_cond = batch.get("x_cond", x1)
        x1_time_cond = batch.get("x_time_cond", x1)
        x1_time_cond_additional = batch.get("x_time_cond_additional", None)

        x1_cond = x1_cond.to(device)
        x1_time_cond = x1_time_cond.to(device)
        x1_time_cond_additional = x1_time_cond_additional.to(
            device) if x1_time_cond_additional is not None else None

        if self.time_transform is not None:
            x1_time_cond = self.time_transform(x1_time_cond)

        margin = batch.get("margin", 0)

        return x1, x1_cond, x1_time_cond, x1_time_cond_additional, margin

    def sample_prior(self, x0_shape):
        return torch.randn(x0_shape).to(self.device)

    def save_model(self, model_dir):
        if self.use_ema:
            with self.ema.average_parameters():
                d = {
                    "model_state": {
                        k: v
                        for k, v in self.state_dict().items()
                        if "emb_model" not in k
                    },
                    "opt_state": self.opt.state_dict()
                }

                torch.save(
                    d, model_dir + "/checkpoint" + str(self.step) + "_EMA.pt")
        else:
            d = {
                "model_state": {
                    k: v
                    for k, v in self.state_dict().items()
                    if "emb_model" not in k
                },
                "opt_state": self.opt.state_dict()
            }

            torch.save(d, model_dir + "/checkpoint" + str(self.step) + ".pt")

    def simclr_loss(self, a, b):

        if len(a.shape) == 3:
            bsize, c, t = a.shape
            a, b = rearrange(a, 'b c t -> (b t) c'), rearrange(
                b, 'b c t -> (b t) c')
        else:
            bsize = a.shape[0]
            t = 1

        sii = torch.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-8)

        atot = a.repeat_interleave(a.shape[0],
                                   dim=0).reshape(a.shape[0], a.shape[0],
                                                  a.shape[1])
        btot = b.repeat(b.shape[0], 1, 1).reshape(a.shape[0], a.shape[0],
                                                  a.shape[1])

        sij = torch.nn.functional.cosine_similarity(atot, btot, dim=2)

        mask = (1 - torch.eye(a.shape[0], a.shape[0])).bool().to(a)

        loss_ii = -torch.log(
            torch.exp(sii) / torch.sum(mask * torch.exp(sij), dim=1))

        loss_ii = loss_ii.reshape(bsize, t)
        return loss_ii

    def ema_loss_balance(self, losses, keys, decay=0.999):
        out = []
        for loss, key in zip(losses, keys):

            self.ema_loss_averages[key] = loss.item() * (
                1 - decay) + self.ema_loss_averages.get(
                    key, loss.item()) * self.ema.decay

            out.append(loss / (self.ema_loss_averages[key] + 1e-8))
        return out

    @gin.configurable
    def fit(self,
            dataloader,
            validloader,
            restart_step,
            model_dir,
            max_steps,
            lr,
            adversarial_weight=0.0,
            adversarial_warmup=10000,
            adversarial_loss="cosine",
            timbre_warmup=100000,
            stop_training_encoder_step=1e10,
            shuffle_zsem=False,
            cycle_start_step=1e10,
            cycle_consistency=False,
            cycle_weights=[0., 0.],
            cycle_mode="interpolant",
            cycle_loss_type="cosine",
            cycle_swap_target="cond",
            drop_targets="both",
            steps_valid=5000,
            steps_display=100,
            steps_save=25000,
            train_encoder=True,
            train_encoder_time=True,
            use_ema=True,
            update_classifier_every=2,
            load_encoders=[True, True, True],
            use_balancer=False,
            **kwargs):

        print("update_classifier_every", update_classifier_every)

        self.train_encoder = train_encoder
        self.train_encoder_time = train_encoder_time
        self.use_ema = use_ema
        self.max_steps = max_steps

        if cycle_loss_type == "full_simclr":
            self.cycle_emb = SimCLREmbedding(
                channels=self.encoder.out_channels,
                time_channels=self.encoder_time.out_channels,
                device=self.device)
        else:
            self.cycle_emb = None
        self.init_train(lr=lr,
                        dataloader=validloader,
                        train_encoders=[train_encoder, train_encoder_time])

        if restart_step > 0:
            state_dict = torch.load(f"{model_dir}/checkpoint" +
                                    str(restart_step) + "_EMA.pt",
                                    map_location="cpu")

            state_dict_model = {
                key: value
                for key, value in state_dict["model_state"].items()
                if (load_encoders[0] or "encoder." not in key) and (
                    load_encoders[1] or "encoder_time" not in key)
            }

            state_dict_model = {
                key: value
                for key, value in state_dict_model.items()
                if (load_encoders[2] or "net." not in key)
            }

            self.load_state_dict(state_dict_model, strict=False)

            try:
                self.opt.load_state_dict(state_dict["opt_state"])
            except Exception as e:
                print(e)
                print("Could not load optimizer state")
            self.step = restart_step + 1

            print("Restarting from step ", self.step)

        if self.use_ema:
            params = list(self.net.parameters())
            self.ema = ExponentialMovingAverage(params, decay=0.999)

        # Setup training of encoders
        if self.train_encoder == False:
            self.encoder.eval()
        if self.train_encoder_time == False:
            self.encoder_time.eval()

        # Loging
        logger = SummaryWriter(log_dir=model_dir + "/logs")
        self.tepoch = tqdm(total=max_steps, unit="batch")

        n_epochs = max_steps // len(dataloader) + 1
        losses_sum = {}
        losses_sum_count = {}

        with open(os.path.join(model_dir, "config.gin"), "w") as config_out:
            config_out.write(gin.operative_config_str())

        if self.step > stop_training_encoder_step:
            print("detaching encoder")
            for param in self.encoder.parameters():
                param.requires_grad = False

        for e in range(n_epochs):
            for batch in dataloader:
                if self.step == stop_training_encoder_step:
                    print("detaching encoder")
                    for param in self.encoder.parameters():
                        param.requires_grad = False

                x1, x1_cond, x1_time_cond, x1_time_cond_additional, margin = self.prep_data(
                    batch, device=self.device)

                if shuffle_zsem:
                    for n in range(x1_cond.shape[0]):
                        zsplit = x1_cond[n].split(2, dim=-1)
                        zsplit = [
                            zsplit[i] for i in torch.randperm(len(zsplit))
                        ]
                        x1_cond[n] = torch.cat(zsplit, dim=-1)

                if self.step > stop_training_encoder_step or not train_encoder:
                    with torch.no_grad():
                        cond = self.encoder(
                            x1_cond) if self.encoder is not None else None
                else:
                    cond = self.encoder(
                        x1_cond) if self.encoder is not None else None

                if self.post_encoder is not None:
                    cond = self.post_encoder(cond)

                time_cond = self.encoder_time(
                    x1_time_cond) if self.encoder_time is not None else None

                time_cond, _ = self.vector_quantizer(
                    time_cond) if self.vector_quantizer is not None else (
                        time_cond, None)

                time_cond = time_cond[..., margin:]

                if self.drop_rate > 0:
                    cond_drop, time_cond_drop = self.cfgdrop(
                        [cond, time_cond],
                        bsize=x1.shape[0],
                        drop_targets=[0, 1] if drop_targets == "both" else [1],
                        drop_rate=self.drop_rate)

                if x1_time_cond_additional is not None:
                    x1_time_cond_additional = self.cfgdrop(
                        [x1_time_cond_additional],
                        bsize=x1.shape[0],
                        drop_targets=[0],
                        drop_rate=0.4)[0]

                    time_cond_drop = torch.cat(
                        [time_cond_drop, x1_time_cond_additional], dim=1)

                # Adversarial step
                if self.step > timbre_warmup and not (
                        self.step % update_classifier_every
                        == 0) and self.classifier is not None:
                    if adversarial_loss == "cosine":
                        classifier_loss = (
                            1 - torch.nn.functional.cosine_similarity(
                                self.classifier(time_cond.detach()),
                                cond.detach(),
                                dim=1,
                                eps=1e-8)).mean()

                    elif adversarial_loss == "mse":
                        classifier_loss = torch.nn.functional.mse_loss(
                            self.classifier(time_cond.detach()),
                            cond.detach(),
                            reduction='mean')

                    self.opt_classifier.zero_grad()
                    classifier_loss.backward()

                    self.opt_classifier.step()

                    lossdict = {
                        "Classifier loss": classifier_loss.item(),
                    }

                # Diffusion step
                else:

                    if self.step < timbre_warmup:
                        time_cond_drop = self.drop_value * torch.ones_like(
                            time_cond_drop)

                    if self.step > timbre_warmup and self.classifier is not None:
                        if adversarial_loss == "cosine":
                            classifier_loss = (
                                1 - torch.nn.functional.cosine_similarity(
                                    self.classifier(time_cond),
                                    cond.detach(),
                                    dim=1,
                                    eps=1e-8)).mean()
                        elif adversarial_loss == "mse":
                            classifier_loss = torch.nn.functional.mse_loss(
                                self.classifier(time_cond),
                                cond.detach(),
                                reduction='mean')

                        adversarial_weight_cur = min(
                            adversarial_weight * (self.step - timbre_warmup) /
                            (adversarial_warmup), adversarial_weight)

                    else:
                        classifier_loss = torch.tensor(0.)
                        adversarial_weight_cur = 0.

                    diffusion_loss, x_transfer, time_cond_target, cond_target = self.diffusion_step(
                        x1,
                        time_cond=time_cond_drop,
                        cond=cond_drop,
                        time_cond_true=time_cond,
                        cond_true=cond,
                        cycle_consistency=cycle_consistency
                        and self.step > cycle_start_step,
                        cycle_mode=cycle_mode,
                        cycle_swap_target=cycle_swap_target)

                    if cycle_consistency and self.step > cycle_start_step:

                        cond_rec = self.encoder(x_transfer)
                        if self.post_encoder is not None:
                            cond_rec = self.post_encoder(cond_rec)

                        time_cond_rec = self.encoder_time(x_transfer)
                        time_cond_rec, _ = self.vector_quantizer(
                            time_cond_rec
                        ) if self.vector_quantizer is not None else (
                            time_cond_rec, None)

                        if cycle_loss_type == "cosine":
                            cond_cycle_loss = (
                                1 - torch.nn.functional.cosine_similarity(
                                    cond_rec,
                                    cond_target.detach(),
                                    dim=1,
                                    eps=1e-8)).mean()

                            time_cond_cycle_loss = (
                                1 - torch.nn.functional.cosine_similarity(
                                    time_cond_rec,
                                    time_cond_target.detach(),
                                    dim=1,
                                    eps=1e-8)).mean()

                        elif cycle_loss_type == "simclr":
                            cond_cycle_loss = self.simclr_loss(
                                cond_rec, cond_target.detach()).mean()

                            time_cond_cycle_loss = self.simclr_loss(
                                time_cond_rec,
                                time_cond_target.detach()).mean()

                        elif cycle_loss_type == "full_simclr":

                            cond_target_emb, cond_rec_emb = self.cycle_emb.encode(
                                cond_target.detach()), self.cycle_emb.encode(
                                    cond_rec)

                            time_cond_perm_emb, time_cond_rec_emb = self.cycle_emb.encode_time(
                                time_cond_target.detach(
                                )), self.cycle_emb.encode_time(time_cond_rec)

                            cond_cycle_loss = self.simclr_loss(
                                cond_target_emb, cond_rec_emb).mean()
                            #
                            time_cond_cycle_loss = self.simclr_loss(
                                time_cond_perm_emb, time_cond_rec_emb).mean()

                    else:
                        cond_cycle_loss = torch.tensor(0.)
                        time_cond_cycle_loss = torch.tensor(0.)

                    lossdict = {
                        "Diffusion loss": diffusion_loss.item(),
                        "Classifier loss": classifier_loss.item(),
                        "Regularisation weight": adversarial_weight_cur,
                        "Cycle loss - cond": cond_cycle_loss.item(),
                        "Cycle loss - time_cond": time_cond_cycle_loss.item(),
                    }

                    if use_balancer:
                        diffusion_loss, classifier_loss, cond_cycle_loss, time_cond_cycle_loss = self.ema_loss_balance(
                            [
                                diffusion_loss, classifier_loss,
                                cond_cycle_loss, time_cond_cycle_loss
                            ], [
                                "Diffusion loss", "Classifier loss",
                                "Cycle loss - time_cond", "Cycle loss - cond"
                            ])

                    loss = diffusion_loss - adversarial_weight_cur * classifier_loss + cycle_weights[
                        0] * cond_cycle_loss + cycle_weights[
                            1] * time_cond_cycle_loss

                    self.opt.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                    self.opt.step()

                if self.use_ema:
                    self.ema.update()

                for k in lossdict:
                    losses_sum[k] = losses_sum.get(k, 0.) + lossdict[k]
                    losses_sum_count[k] = losses_sum_count.get(k, 0) + 1

                if self.step % steps_display == 0:
                    self.tepoch.set_postfix(loss=losses_sum["Diffusion loss"] /
                                            steps_display)
                    for k in losses_sum:
                        logger.add_scalar('Loss/' + k,
                                          losses_sum[k] /
                                          max(1, losses_sum_count[k]),
                                          global_step=self.step)
                        losses_sum[k] = 0.
                        losses_sum_count[k] = 0

                if self.step % steps_valid == 20:
                    with torch.no_grad() and self.ema.average_parameters():

                        ## VALIDATION

                        lossval = {}

                        for i, batch in enumerate(validloader):
                            x1, x1_cond, x1_time_cond, x1_time_cond_additional, margin = self.prep_data(
                                batch, device=self.device)

                            cond = self.encoder(
                                x1_cond) if self.encoder is not None else None

                            cond = self.post_encoder(
                                cond
                            ) if self.post_encoder is not None else cond

                            if self.encoder_time is not None:
                                time_cond = self.encoder_time(x1_time_cond)
                                time_cond, _ = self.vector_quantizer(
                                    time_cond
                                ) if self.vector_quantizer is not None else (
                                    time_cond, None)
                                time_cond = time_cond[..., margin:]
                                time_cond = torch.cat(
                                    (time_cond, x1_time_cond_additional),
                                    dim=1
                                ) if x1_time_cond_additional is not None else time_cond
                            else:
                                time_cond = None

                            if self.step < timbre_warmup:
                                time_cond = self.drop_value * torch.ones_like(
                                    time_cond)

                            diffusion_loss, _, _, _ = self.diffusion_step(
                                x1, time_cond=time_cond, cond=cond)

                            lossdict = {
                                "Diffusion loss": diffusion_loss.item()
                            }

                            if self.classifier is not None:
                                if adversarial_loss == "cosine":
                                    classifier_loss = (
                                        1 -
                                        torch.nn.functional.cosine_similarity(
                                            self.classifier(time_cond),
                                            cond.detach(),
                                            dim=1,
                                            eps=1e-8)).mean()

                                elif adversarial_loss == "mse":
                                    classifier_loss = torch.nn.functional.mse_loss(
                                        self.classifier(time_cond),
                                        cond.detach(),
                                        reduction='mean')

                                lossdict[
                                    "Classifier loss"] = classifier_loss.item(
                                    )

                            for k in lossdict:
                                lossval[k] = lossval.get(k, 0.) + lossdict[k]

                            if i == 100:
                                break

                        for k in lossval:
                            logger.add_scalar('Loss/valid/' + k,
                                              lossval[k] / 100,
                                              global_step=self.step)

                        ## SAMPLING
                        x1 = x1[:6].to(self.device)
                        time_cond = time_cond[:6] if time_cond is not None else None
                        cond = cond[:6] if cond is not None else None
                        x0 = self.sample_prior(x1.shape)

                        audio_true = self.emb_model.decode(x1.cpu()).cpu()

                        for i in range(x1.shape[0]):
                            logger.add_audio("true/" + str(i),
                                             audio_true[i],
                                             global_step=self.step,
                                             sample_rate=self.sr)

                        for nb_steps in [1, 5, 20, 40]:
                            x1_rec = self.sample(x0,
                                                 nb_steps=nb_steps,
                                                 time_cond=time_cond,
                                                 cond=cond)

                            audio_rec = self.emb_model.decode(
                                x1_rec.cpu()).cpu()

                            for i in range(x1.shape[0]):
                                logger.add_audio("generated/" + str(nb_steps) +
                                                 "steps/" + str(i),
                                                 audio_rec[i],
                                                 global_step=self.step,
                                                 sample_rate=self.sr)

                if self.step % steps_save == 0:
                    self.save_model(model_dir)

                self.tepoch.update(1)
                self.step += 1


class RectifiedFlow(Base):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def diffusion_step(self,
                       x1,
                       time_cond,
                       cond,
                       time_cond_true=None,
                       cond_true=None,
                       cycle_consistency=False,
                       cycle_mode="interpolant",
                       cycle_swap_target="cond"):

        x0 = torch.randn_like(x1)

        target = x1 - x0
        t = torch.rand(x0.size(0), 1, 1).to(self.device)

        if cycle_mode == "x0" and cycle_consistency:
            t = torch.zeros_like(t)

        interpolant = (1 - t) * x0 + t * x1

        model_output = self.net(interpolant,
                                time_cond=time_cond,
                                cond=cond,
                                time=t)

        loss = ((model_output - target)**2).mean()

        if cycle_consistency:
            if time_cond_true is None:
                time_cond_true = time_cond
            if cond_true is None:
                cond_true = cond

            if cycle_swap_target == "time_cond":
                time_cond_target = time_cond_true[torch.randperm(
                    time_cond_true.shape[0])]
                cond_target = cond_true
            elif cycle_swap_target == "cond":
                time_cond_target = time_cond_true
                cond_target = cond_true[torch.randperm(cond_true.shape[0])]

            elif cycle_swap_target == "alternate":
                n = time_cond_true.shape[0]
                indices = torch.randperm(time_cond_true.shape[0])

                time_cond_target = time_cond_true.clone()
                time_cond_target[indices[:n //
                                         2]] = time_cond_true[indices[n // 2:]]

                cond_target = cond_true.clone()
                cond_target[indices[n // 2:]] = cond_true[indices[:n // 2]]

            if cycle_mode == "interpolant":
                model_output_transfer = self.net(interpolant,
                                                 time=t,
                                                 time_cond=time_cond_target,
                                                 cond=cond_target)
                x_transfer = interpolant + t * model_output_transfer

            elif cycle_mode == "x0":
                x0 = torch.randn_like(x0)
                model_output_transfer = self.net(x0,
                                                 time=torch.zeros_like(t),
                                                 time_cond=time_cond_target,
                                                 cond=cond_target)
                x_transfer = x0 + model_output_transfer

            else:
                raise ValueError("Invalid cycle mode : " + cycle_mode)

        else:
            x_transfer = None
            time_cond_target = None
            cond_target = None

        return loss, x_transfer, time_cond_target, cond_target

    def model_forward(self,
                      x,
                      time,
                      cond,
                      time_cond,
                      guidance_cond_factor,
                      guidance_joint_factor,
                      total_guidance,
                      time_cond_add=None):

        full_time = time.repeat(4, 1, 1)
        full_x = x.repeat(4, 1, 1)
        full_cond = torch.cat([
            cond, self.drop_value * torch.ones_like(cond),
            self.drop_value * torch.ones_like(cond), cond
        ])
        full_time_cond = torch.cat([
            time_cond, self.drop_value * torch.ones_like(time_cond), time_cond,
            self.drop_value * torch.ones_like(time_cond)
        ])

        if time_cond_add is not None:
            time_cond_add = time_cond_add.repeat(4, 1, 1)
            full_time_cond = torch.cat([full_time_cond, time_cond_add], dim=1)

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

    @torch.no_grad()
    def sample(self,
               x0,
               cond,
               time_cond,
               nb_steps,
               guidance_cond_factor=0.,
               guidance_joint_factor=1.,
               total_guidance=1.,
               return_trajectory=False,
               time_cond_add=None):
        dt = 1 / nb_steps
        t_values = torch.linspace(0, 1, nb_steps + 1).to(self.device)[:-1]
        x = x0.to(self.device)

        x_out = []
        for t in t_values:
            t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
            x = x + self.model_forward(
                x, t, cond, time_cond, guidance_cond_factor,
                guidance_joint_factor, total_guidance, time_cond_add) * dt
            x_out.append(x.cpu())

        if return_trajectory:
            return x, torch.stack(x_out, dim=-1)
        return x


class EDM(Base):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p_mean, self.p_std = -1.2, 1.2

    def _get_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2

    def _get_scalings(self, sigma: torch.Tensor):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 +
                                           self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = 0.25 * sigma.log()

        return c_skip, c_out, c_in, c_noise

    def model_forward(self, x, sigma, cond, time_cond):

        c_skip, c_out, c_in, c_noise = self._get_scalings(sigma)

        f_xy = self.net(c_in[:, None, None] * x,
                        time=c_noise,
                        cond=cond,
                        time_cond=time_cond)

        d = c_skip[:, None, None] * x + c_out[:, None, None] * f_xy
        return d

    def diffusion_step(self, x: torch.Tensor, cond: torch.Tensor,
                       time_cond: torch.Tensor, **kwargs) -> torch.Tensor:
        b, *_ = x.shape

        z = torch.randn(b, 1, 1, device=self.device)
        sigma = (z * self.p_std + self.p_mean).exp()

        x_noisy = x + torch.randn_like(x) * sigma

        d = self.model_forward(x_noisy,
                               sigma.view(-1),
                               cond=cond,
                               time_cond=time_cond)
        weight = self._get_weight(sigma)

        loss = weight * ((d - x)**2)
        loss = loss.mean()

        return loss

    def get_steps(self, n_steps, rho, sigma_min, sigma_max):
        step_indices = torch.arange(0, n_steps)
        t_steps = (sigma_min**(1 / rho) + step_indices / (n_steps - 1) *
                   (sigma_max**(1 / rho) - sigma_min**(1 / rho)))**rho
        t_steps = torch.cat([torch.tensor([0.]), t_steps])
        t_steps = torch.as_tensor(t_steps)

        return t_steps

    @torch.no_grad()
    def sample(self,
               x0,
               cond,
               time_cond,
               nb_steps,
               sigma_min=0.01,
               sigma_max=80,
               rho=7,
               return_trajectory=False,
               second_order=False):

        out_samples = []

        # Sample initial noise
        samples = x0.to(self.device) * sigma_max

        # sample steps
        t_steps = self.get_steps(nb_steps, rho, sigma_min,
                                 sigma_max).to(self.device)

        # Perform the sampling
        for i in range(nb_steps - 1):
            sigma_t = t_steps[nb_steps - i - 1]
            sigma_t_next = t_steps[nb_steps - i - 2]

            model_output = self.model_forward(samples,
                                              sigma_t.view(-1).repeat(
                                                  x0.shape[0]),
                                              cond=cond,
                                              time_cond=time_cond)

            d_cur = (samples - model_output) / sigma_t
            samples_next = samples + d_cur * (sigma_t_next - sigma_t)

            if i < nb_steps - 2 and second_order:
                model_output = self.model_forward(samples_next,
                                                  sigma_t_next.view(-1).repeat(
                                                      x0.shape[0]),
                                                  cond=cond,
                                                  time_cond=time_cond)
                d_prime = (samples_next - model_output) / sigma_t_next
                samples_next = samples + (sigma_t_next - sigma_t) * (
                    0.5 * d_prime + 0.5 * d_cur)

            out_samples.append(samples_next.cpu())
            samples = samples_next.clone()

        if return_trajectory:
            return samples, torch.stack(out_samples, dim=-1)

        return samples


class sCM(Base):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p_mean, self.p_std = -1.2, 1.2
        self.warmup = 20000
        self.c = .1
        self.extra_modules = nn.ModuleDict({
            "w_phi":
            nn.Linear(self.net.time_channels, 1)
        }).to(self.device)

    def diffusion_step(self, x: torch.Tensor, cond: torch.Tensor,
                       time_cond: torch.Tensor, **kwargs) -> torch.Tensor:

        b = x.shape[0]
        sigma = torch.randn(b, 1, 1,
                            device=self.device) * self.p_std + self.p_mean

        sigma = sigma.exp()
        t = torch.arctan(sigma / self.sigma_data)

        # Get z and x_t
        z = torch.randn_like(x) * self.sigma_data
        x_t = torch.cos(t) * x + torch.sin(t) * z

        # consistency trainin
        dx_dt = t.cos() * z - t.sin() * x

        # Compute model
        t_emb = self.net.time_emb(t.view(-1))
        #f = self.model(x_t / self.sigma_data, t_emb)

        # Compute JVP
        fn = lambda x_t, tc: self.net(x_t,
                                      time_emb=self.net.time_emb(tc.view(-1)),
                                      cond=cond,
                                      time_cond=time_cond)

        v_x = torch.cos(t) * torch.sin(t) * dx_dt / self.sigma_data
        v_t = torch.cos(t) * torch.sin(t)
        f_theta, f_theta_grad = torch.func.jvp(fn, (x_t / self.sigma_data, t),
                                               (v_x, v_t))

        f_theta_minus = f_theta.clone().detach()
        f_theta_grad = f_theta_grad.detach()

        r = min(1., self.step / self.warmup)
        g = -torch.cos(t) * torch.cos(t) * (self.sigma_data * f_theta_minus -
                                            dx_dt)
        second_term = -r * (torch.cos(t) * torch.sin(t) * x_t +
                            self.sigma_data * f_theta_grad)
        g = g + second_term

        # G NORMALIZATION
        g_norm = torch.linalg.vector_norm(g, dim=(1, 2), keepdim=True)
        g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())
        g = g / (g_norm + self.c)

        # WEIGHTING
        weight = 1 / sigma
        w = self.extra_modules["w_phi"](t_emb)
        loss = (f_theta - f_theta_minus - g)**2

        loss = loss * weight * (1 / w.exp()) + w
        loss = loss.mean()

        return loss

    @torch.no_grad()
    def sample(self,
               x0,
               cond,
               time_cond,
               nb_steps=50,
               return_trajectory=False):
        xout = []

        # sample noise
        xn = x0 * self.sigma_data

        # Linear sampling
        t = torch.linspace(np.pi / 2, 0, nb_steps + 1)

        for i, tc in enumerate(t[:-1]):
            tc = tc.view(-1).to(self.device)
            tc_next = t[i + 1].view(-1).to(self.device)
            delta_t = tc_next - tc

            xn = xn * torch.cos(delta_t) + self.sigma_data * torch.sin(
                delta_t) * self.net(xn / self.sigma_data,
                                    time=tc.repeat(xn.shape[0], 1),
                                    cond=cond,
                                    time_cond=time_cond)

            xout.append(xn.clone().cpu())

        if return_trajectory:
            return xn, torch.stack(xout, dim=-1)
        return xn


class SimCLREmbedding(nn.Module):

    def __init__(self, channels, time_channels, emb_dim=32, device="cpu"):
        super().__init__()

        self.emb_dim = emb_dim

        self.projector = nn.Sequential(nn.Linear(channels, emb_dim), nn.ReLU(),
                                       nn.Linear(emb_dim, emb_dim))
        self.projector_time = nn.Sequential(nn.Linear(time_channels, emb_dim),
                                            nn.ReLU(),
                                            nn.Linear(emb_dim, emb_dim))

        self.to(device)

    def encode(self, x):
        return self.projector(x) / torch.linalg.norm(
            self.projector(x), dim=-1, keepdim=True)

    def encode_time(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.projector_time(x)
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
        return x.permute(0, 2, 1)
