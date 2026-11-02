# Copyright (c) CryoJDM
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Core."""

from __future__ import annotations

import copy
import csv
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from monai.data import CacheDataset, Dataset, ThreadDataLoader
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, SegResNet, SwinUNETR, UNet
from monai.optimizers import Novograd
from monai.transforms import Compose, Lambda
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_info
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from cryojdm.utils import (
    ApplyFourier,
    ConfigHandler,
    DataHandler,
    FileHandler,
    LoadTomo,
    Mask3D,
    MetricsCalculator,
    RandTrans,
)

sys.path.append(str(Path(__file__).parent / "third_party"))
from models.em_net_model_ours import EMNet
from models.ours import JDMNet
from models.segmamba import SegMamba
from models.umamba import UMambaBot
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR, WarmupCosineSchedule

warnings.filterwarnings("ignore")


global_dict = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "novograd": Novograd,
    "LinearWarmupCosineAnnealingLR": LinearWarmupCosineAnnealingLR,
    "WarmupCosineSchedule": WarmupCosineSchedule,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "StepLR": StepLR,
    "ours": JDMNet,
    "emnet": EMNet,
    "umamba": UMambaBot,
    "segmamba": SegMamba,
    "segresnet": SegResNet,
    "unet": UNet,
    "unetr": UNETR,
    "swinunetr": SwinUNETR,
}


def get_transforms(cfgs: DictConfig) -> Dict[str, Compose]:
    """Create data transformation pipelines for training, validation, and inference."""

    def add_epoch(d: Dict, train=True) -> Dict:
        """Update the 'epoch' field in the input sample dictionary."""
        d["epoch"] = d.get("epoch", -1) + 1 if train else 0
        return copy.deepcopy(d)

    add_epoch_train = partial(add_epoch, train=True)
    add_epoch_val = partial(add_epoch, train=False)

    # Operational mode: 0 for our method, any other value for Noise2Noise (N2N) training.
    train_keys = ["all"] if cfgs.mode == 0 else ["all", "even", "odd"]

    # Build validation and inference keys from train_keys and infer_maps
    infer_maps = {1: [], 2: ["label"], 3: ["label", "clean"]}
    val_keys = train_keys + infer_maps[cfgs.infer_mode]
    infer_keys = ["all"] + infer_maps[cfgs.infer_mode]

    # Assign tensor data types to each key (e.g., label as int8, others as float32)
    type_maps = {"label": torch.int8}
    val_types = [type_maps.get(k, torch.float32) for k in val_keys]
    infer_types = [type_maps.get(k, torch.float32) for k in infer_keys]

    # Common transformation parameters shared across training/validation
    common_transforms = {
        "roi_len": cfgs.roi_len,
        "max_epochs": cfgs.max_epochs,
        "dataset_len": cfgs.dataset_len,
        "tilt_angle": cfgs.tilt_angle,
        "mw_factor": float(cfgs.mw_factor),
        "use_blind": cfgs.use_blind,
        "seed": cfgs.seed,
    }

    train_transform = Compose(
        [
            Lambda(func=add_epoch_train),
            LoadTomo(keys=train_keys, add_channel=True),
            RandTrans(**common_transforms),
        ]
    )

    val_transform = Compose(
        [
            Lambda(func=add_epoch_val),
            LoadTomo(keys=val_keys, add_channel=True, output_type=val_types),
            RandTrans(**common_transforms),
        ]
    )

    infer_transform = Compose(
        [
            LoadTomo(keys=infer_keys, add_channel=True, output_type=infer_types),
            # NormalizeIntensityd(keys=["all"], nonzero=False),
        ]
    )
    return {"train": train_transform, "val": val_transform, "infer": infer_transform}


class LitDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling dataset preparation."""

    def __init__(self, transforms: Dict, cfgs: DictConfig) -> None:
        """Initialize the DataModule with transforms and configurations."""
        super().__init__()
        self.train_transforms = transforms["train"]
        self.val_transforms = transforms["val"]
        self.infer_transforms = transforms["infer"]

        self.json_path = cfgs.json_path
        self.num_workers = cfgs.num_workers
        self.batch_size = cfgs.batch_size

        # Prepare the dataset file paths.
        files = FileHandler.load_json(self.json_path)
        self.train_files = files["train"]
        self.val_files = files["val"]
        self.infer_files = files["infer"]

    def train_dataloader(self, cache: bool = False) -> ThreadDataLoader:
        """Return a DataLoader for training dataset."""
        _dataset = (
            partial(CacheDataset, cache_rate=1, num_workers=self.num_workers) if cache else Dataset
        )
        train_dataset = _dataset(self.train_files, transform=self.train_transforms)
        return ThreadDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=cache,
            use_thread_workers=True,
            # sampler=DistributedSampler(train_dataset, shuffle=False),
        )

    def val_dataloader(self) -> ThreadDataLoader:
        """Return a DataLoader for validation dataset."""
        val_dataset = CacheDataset(
            self.val_files,
            transform=self.val_transforms,
            cache_rate=1,
            num_workers=self.num_workers,
        )
        return ThreadDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            use_thread_workers=True,
            # sampler=DistributedSampler(val_dataset, shuffle=False),
        )

    def test_dataloader(self) -> ThreadDataLoader:
        """Return a DataLoader for inference dataset."""
        infer_dataset = CacheDataset(
            self.infer_files,
            transform=self.infer_transforms,
            cache_rate=1,
            num_workers=self.num_workers,
        )
        return ThreadDataLoader(
            infer_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            use_thread_workers=True,
            # sampler=DistributedSampler(infer_dataset, shuffle=False),
        )


def get_latest_version(folder: Union[str, Path]) -> int:
    """Return the largest 'version_xxx' index, or -1 if none exists."""
    paths = sorted(Path(folder).glob("version_???"))
    if paths:
        return int(paths[-1].stem.split("version_")[-1])
    else:
        return -1


def get_trainer(name: str, cfgs: DictConfig) -> pl.Trainer:
    """Create and configure a PyTorch Lightning Trainer instance."""
    # Set custom checkpoint naming characters
    pl.callbacks.ModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"
    pl.callbacks.ModelCheckpoint.CHECKPOINT_JOIN_CHAR = "_"

    # Compute next version directory
    latest_version = get_latest_version(cfgs.output_dir / name)
    version = f"version_{latest_version + 1:03d}"

    # Device and strategy configuration
    if cfgs.devices == "cpu":
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
    else:
        accelerator = "gpu"
        devices = cfgs.devices
        strategy = pl.strategies.DDPStrategy(
            find_unused_parameters=(cfgs.model in ["umamba", "unetr"])
        )

    # Inference-only trainer
    if name == "infer":
        return pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            logger=False,
            enable_checkpointing=False,
            log_every_n_steps=0,
            inference_mode=True,
        )

    # Setup loggers
    loggers = [
        pl.loggers.CSVLogger(
            save_dir=cfgs.output_dir,
            name=name,
            version=version,
        ),
        pl.loggers.TensorBoardLogger(
            save_dir=cfgs.output_dir,
            name=name,
            version=version,
        ),
    ]

    # Setup callbacks
    callbacks = [
        # Save best top-k checkpoints based on validation loss.
        pl.callbacks.ModelCheckpoint(
            dirpath=f"{loggers[0].log_dir}/checkpoints/top_k",
            filename="{epoch:05d}_{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
            verbose=True,
            save_top_k=3,
        ),
        # Save checkpoint every epoch without monitoring a metric.
        pl.callbacks.ModelCheckpoint(
            dirpath=f"{loggers[0].log_dir}/checkpoints/epoch",
            filename="{epoch:05d}",
            monitor="epoch",
            verbose=True,
            save_top_k=-1,
            every_n_epochs=1,
        ),
    ]

    return pl.Trainer(
        max_epochs=cfgs.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        check_val_every_n_epoch=cfgs.check_val_every_n_epoch,
        deterministic=True,
        logger=loggers,
        callbacks=callbacks,  # type: ignore[arg-type]
        num_sanity_val_steps=0,
    )


class LitNet(pl.LightningModule):
    """A PyTorch Lightning module with custom configurations."""

    def __init__(self, cfg_model: DictConfig, cfg_basic: DictConfig) -> None:
        """Initialize with model and basic configurations."""
        super().__init__()
        self.save_hyperparameters(cfg_model)
        self.cfg_basic = cfg_basic

        # Initialize optimizer, scheduler, and network model
        self.optim = partial(global_dict[cfg_model.optimizer], **cfg_model.optim)
        self.sched = (
            partial(global_dict[cfg_model.scheduler], **cfg_model.sched)
            if cfg_model.scheduler
            else None
        )

        model_name = cfg_basic.model
        self.net = global_dict[model_name](**ConfigHandler.convert(cfg_model.net))
        if model_name in ["segresnet", "umamba"]:
            torch.use_deterministic_algorithms(False)

        self.mode = cfg_basic.mode
        self.sw_batch_size = cfg_basic.sw_batch_size or cfg_basic.batch_size * 3
        self.roi_size = [cfg_basic.roi_len] * cfg_basic.spatial_dims

        self.mask_sphere = Mask3D.Sphere(shape=self.roi_size)
        self.mask_mw = Mask3D.MissingWedge(
            shape=self.roi_size,
            mw_angle=180 - cfg_basic.tilt_angle * 2,
        )

        self.save_mrc = partial(FileHandler.save_mrc, voxel_size=cfg_basic.voxel_size)
        self.fft_func = ApplyFourier.fft_3d
        self.ifft_func = ApplyFourier.ifft_3d
        self.fmask_func = ApplyFourier.apply_fourier_mask_to_tomo
        self.norm_func = partial(DataHandler.scale_tensor, norm=True)
        self.scale_func = partial(DataHandler.scale_tensor, scale=True)
        self.expand_func = (
            lambda x, times: x.unsqueeze(0).unsqueeze(0).expand(times, -1, -1, -1, -1)
        )
        self.crop_func = partial(DataHandler.crop_center, target_shape=self.roi_size)

        self.infer_func = partial(
            sliding_window_inference,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self._predictor,
            overlap=0.8,  # cfg_basic.infer_overlap,
            padding_mode="replicate",
        )
        self.log_func = partial(
            self.log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # loss
        self.mse_loss_func = torch.nn.MSELoss(reduction="mean")
        self.use_wloss = cfg_basic.use_wloss
        self.best_val_loss = float("inf")

        # metrics
        self.save_metrics = True
        self.dataset_name = (
            str(self.cfg_basic.train_dir).split("dataset/")[-1].split("/2_train")[0]
        )
        self.iterations = None if self.dataset_name in ["DS_10439", "SHREC_2021", "TTS"] else 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass the input tensor through the network model."""
        return self.net(x)

    def configure_optimizers(self) -> Union[Optimizer, Dict]:  # type: ignore[override]
        """Configure optimizers and learning rate schedulers."""
        optimizer = self.optim(self.parameters())
        if not self.sched:
            return optimizer

        scheduler = self.sched(optimizer)
        if self.sched == "ReduceLROnPlateau":
            interval = "epoch"
            frequency = self.cfg_basic.check_val_every_n_epoch
        else:
            interval = "step"
            frequency = 1
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": interval,
                "frequency": frequency,
            },
        }

    def _save(
        self,
        tomo: torch.Tensor,
        suffix: str = "pred",
        save_dir: Optional[str] = None,
    ) -> None:
        """Save the given tensor with a specified suffix."""
        latest_version = get_latest_version(self.cfg_basic.train_dir)
        version = (
            min(f"version_{latest_version:03d}", self.logger.version)
            if getattr(self.logger, "version", None) is not None
            else f"version_{latest_version:03d}"
        )

        save_dir = Path(save_dir) if save_dir else self.cfg_basic.train_dir / version
        save_path = save_dir / f"epoch{self.current_epoch:05d}_{suffix}.mrc"
        self.save_mrc(save_path, tomo.squeeze())

    def sw_func(self, freq: torch.Tensor) -> torch.Tensor:
        """Performs spectral whitening on a given frequency domain tensor.

        magnitude_spectrum = torch.abs(freq)
        phase_spectrum = torch.angle(freq)
        white_magnitude = torch.ones_like(magnitude_spectrum)
        white_freq = white_magnitude * torch.exp(1j * phase_spectrum)

        or

        phase_spectrum = torch.angle(freq)
        white_freq = torch.exp(1j * phase_spectrum)
        """
        eps = 1e-12
        white_freq = freq / (torch.abs(freq) + eps)
        return white_freq

    def _get_spatial_loss(
        self,
        img: torch.Tensor,
        ifft_pred: torch.Tensor,
        mask_comp: torch.Tensor,
        lambda_f: float = 1.0,
        lambda_w: float = 0.0,
    ) -> torch.Tensor:
        """Calculates the Mean Squared Error (MSE) loss between predicted and target images.
        Optionally includes a spectral whitening component if lambda_w is significant.
        """
        # Compute the masked FFT and IFFT components for prediction and target
        fft_pred_comp = self.fft_func(ifft_pred) * mask_comp
        ifft_pred_comp = self.ifft_func(fft_pred_comp).real
        fft_targ_comp = self.fft_func(img) * mask_comp
        ifft_targ_comp = self.ifft_func(fft_targ_comp).real
        loss_f = self.mse_loss_func(ifft_pred_comp, ifft_targ_comp)
        loss = lambda_f * loss_f

        # If lambda_w is significant, add the spectral whitening loss
        if lambda_w < 0.1:
            return loss / lambda_f

        ifft_pred_comp_w = self.ifft_func(self.sw_func(fft_pred_comp) * mask_comp).real
        ifft_targ_comp_w = self.ifft_func(self.sw_func(fft_targ_comp) * mask_comp).real
        loss_w = self.mse_loss_func(ifft_targ_comp_w, ifft_pred_comp_w)
        loss += lambda_w * loss_w

        return loss / (lambda_f + lambda_w)

    def _get_spectral_loss(
        self,
        img: torch.Tensor,
        ifft_pred: torch.Tensor,
        mask_comp: torch.Tensor,
        lambda_f: float = 1.0,
        lambda_w: float = 2.0,
    ) -> torch.Tensor:
        """Calculates a weighted frequency loss between predicted and target images.
        Optionally includes a spectral whitening component if lambda_w is significant.
        """

        def masked_mse_loss(
            diff: torch.Tensor,
            mask: torch.Tensor,
            alpha: int = 2,
        ) -> torch.Tensor:
            """Computes a normalized masked L_alpha loss.

            Args:
                diff (torch.Tensor): Difference tensor (e.g., pred - target), shape [B, ...].
                mask (torch.Tensor): Binary mask of the same shape or broadcastable to diff.
                alpha (int): Power index, e.g., 2 for MSE, 1 for L1 loss.

            Returns:
                torch.Tensor: Scalar masked loss (averaged over batch).
            """
            # Compute element-wise masked error
            loss_masked = torch.abs(diff).pow(alpha) * mask

            # Sum over all spatial/frequency dimensions except batch
            sum_dims = tuple(range(1, diff.ndim))
            mask_sum = torch.clamp(mask.sum(dim=sum_dims), min=1)

            # Mean per sample, then mean over batch
            return (loss_masked.sum(dim=sum_dims) / mask_sum).mean()

        # Compute the masked FFT components for prediction and target
        fft_pred = self.fft_func(ifft_pred)
        fft_targ = self.fft_func(img)
        loss_f = masked_mse_loss(fft_pred - fft_targ, mask_comp)
        loss = lambda_f * loss_f

        # If lambda_w is significant, add the spectral whitening loss
        if lambda_w < 0.1:
            return loss / lambda_f

        fft_pred_w = self.sw_func(fft_pred)
        fft_targ_w = self.sw_func(fft_targ)
        loss_w = masked_mse_loss(fft_pred_w - fft_targ_w, mask_comp)
        loss += lambda_w * loss_w
        return loss / (lambda_f + lambda_w)

    def train_mwc_mode0(self, batch: Dict) -> float:
        """Trains the model with mode 0 settings."""
        img = self.norm_func(batch["all_rot"])
        mask_blind = batch["mask_blind"]
        mask_mw_rot = batch["mask_mw_rot"]
        mask_mw_fix = batch["mask_mw_fix"]
        mask_inte = mask_mw_rot * mask_mw_fix
        mask_comp = mask_mw_rot & ~mask_inte

        img_masked = self.fmask_func(img, mask_inte * mask_blind)

        ifft_pred = self.forward(img_masked)

        if self.use_wloss is None:
            loss = self._get_spatial_loss(
                img,
                ifft_pred,
                mask_comp,
                lambda_f=1.0,
                lambda_w=0.0,
            )
        elif self.use_wloss[0] < 0:
            loss = self._get_spatial_loss(
                img,
                ifft_pred,
                mask_comp,
                lambda_f=abs(float(self.use_wloss[0])),
                lambda_w=abs(float(self.use_wloss[1])),
            )
        else:
            loss = self._get_spectral_loss(
                img,
                ifft_pred,
                mask_comp,
                lambda_f=float(self.use_wloss[0]),
                lambda_w=float(self.use_wloss[1]),
            )

        # if self.current_epoch % 10 == 0 and self.trainer.local_rank == 0:
        #     save_dir = "/data/ldap_shared/home/s_ntx/CryoJDM/demo/figure/rot"
        # blind = 0.01

        #     self._save(
        #         torch.log1p(torch.abs(self.fft_func(img)).pow(2))[0],
        #         "10_rot_mag",
        #         save_dir,
        #     )
        #     self._save(mask_mw_rot[0], "00_mask_mw_rot", save_dir)
        #     self._save(mask_mw_fix[0], "00_mask_mw_fix", save_dir)
        #     self._save(mask_blind[0], "02_mask_blind", save_dir)
        #     self._save(mask_inte[0], "01_mask_inte", save_dir)
        #     self._save(mask_comp[0], "01_mask_comp", save_dir)
        #     self._save((mask_inte * mask_blind)[0], "03_mask_inte_blind", save_dir)

        return loss, ifft_pred

    def train_dnr_mode1(self, batch: Dict) -> float:
        """Trains the model with mode 1 settings."""
        even = self.norm_func(self.crop_func(batch["even"]))  # type: ignore[arg-type]
        odd = self.norm_func(self.crop_func(batch["odd"]))  # type: ignore[arg-type]
        full = self.norm_func(self.crop_func(batch["all"]))

        f_even = self.forward(even)
        loss = self.mse_loss_func(f_even, odd)

        with torch.no_grad():
            f_full = self.forward(full)
        return loss, f_full

    def train_dnr_mode2(self, batch: Dict) -> float:
        """Trains the model with mode 2 settings."""
        mask = batch["mask_mw_rot"]

        even = self.fmask_func(self.norm_func(batch["even_rot"]), mask)
        odd = self.fmask_func(self.norm_func(batch["odd_rot"]), mask)
        full = self.fmask_func(self.norm_func(batch["all_rot"]), mask)

        f_even = self.forward(even)
        f_even = self.fmask_func(f_even, mask)
        loss = self.mse_loss_func(f_even, odd)

        with torch.no_grad():
            f_full = self.forward(full)
        # f_full = self.fmask_func(self.norm_func(f_full), mask)

        return loss, f_full

    def _predictor(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize the input image and applies a Fourier mask if in mode 0 or 2."""
        img = self.norm_func(img)
        b, device = img.shape[0], img.device
        spheres = self.expand_func(self.mask_sphere, b).to(device)

        if self.mode in [0, 2]:
            mask_sphere_mw = self.expand_func(self.mask_mw * self.mask_sphere, b).to(device)
            img = self.fmask_func(img, mask_sphere_mw)
        pred = self.forward(img)
        pred = self.fmask_func(pred, spheres)
        return pred

    def get_loss_pred(self, batch: Dict, add_keys: List = []) -> torch.Tensor:
        """Compute the loss and prediction based on the current mode."""
        img = batch["all"]
        b, device = img.shape[0], img.device

        self.mask_spheres = self.expand_func(self.mask_sphere, b).to(device)
        self.mask_mws = self.expand_func(self.mask_mw, b).to(device)

        if self.mode == 0:
            batch = RandTrans.get_train_batch(
                d=batch,
                mask_sphere=self.mask_spheres,
                mask_mw=self.mask_mws,
                tomo_keys=["all"] + add_keys,
            )
            loss, pred = self.train_mwc_mode0(batch)
        elif self.mode == 1:
            loss, pred = self.train_dnr_mode1(batch)
        elif self.mode == 2:
            batch = RandTrans.get_train_batch(
                d=batch,
                mask_sphere=self.mask_spheres,
                mask_mw=self.mask_mws,
                tomo_keys=["all", "even", "odd"] + add_keys,
            )
            loss, pred = self.train_dnr_mode2(batch)
        else:
            raise ValueError("Unsupported mode. Supported values are 0, 1 or 2.")
        pred = self.fmask_func(pred, self.mask_spheres)
        return loss, pred, batch

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Executes a single training step."""
        loss, *_ = self.get_loss_pred(batch)

        self.log_func("train_loss", loss)
        return loss  # type: ignore[return-value]

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Executes a single validation step."""
        loss, pred, batch = self.get_loss_pred(batch, add_keys=["clean", "label"])
        self.log_func("val_loss", loss)
        self.best_val_loss = min(loss, self.best_val_loss)

        if self.mode == 1:
            clean = self.crop_func(batch["clean"]) if "clean" in batch else None
            label = self.crop_func(batch["label"]) if "label" in batch else None
        else:
            clean = batch.get("clean_rot")
            label = batch.get("label_rot")

        metrics = self._cal_metrics(pred, clean=clean, label=label, iterations=self.iterations)
        for key in metrics.keys() & {"PSNR", "SSIM", "CC", "LFD", "SNR"}:
            self.log_func(key, metrics[key])
        return loss

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        """Executes a single inference step."""
        name = batch["name"][0]
        img = batch["all"]

        pred = self.infer_func(img)
        save_dir = Path(self.cfg_basic.infer_save_dir)
        self.save_mrc(save_dir / f"{name}.mrc", pred.squeeze())

        if self.save_metrics:
            clean = batch.get("clean")
            label = batch.get("label")

            self._save_metrics(
                pred=pred,
                clean=clean,
                label=label,
                name=name,
                dataset=self.dataset_name,
            )
            self._save_metrics(
                pred=img,
                clean=clean,
                label=label,
                name=f"{name}_noisy",
                dataset=self.dataset_name,
            )

            # b, device = img.shape[0], img.device
            # target_shape = [min(img.shape[2:])] * self.cfg_basic.spatial_dims
            # crop_func = partial(DataHandler.crop_center, target_shape=target_shape)
            # mask_sphere = Mask3D.Sphere(shape=target_shape)
            # mask_mw = Mask3D.MissingWedge(
            #     shape=target_shape, mw_angle=180 - self.cfg_basic.tilt_angle * 2
            # )
            # mask_comp = self.expand_func(mask_sphere - mask_mw * mask_sphere, b).to(
            #     device
            # )

            # pred_cm = self.fmask_func(crop_func(pred), mask_comp)
            # clean_cm = self.fmask_func(crop_func(clean), mask_comp)
            # label_c = crop_func(label)

            # self._save_metrics(
            #     pred=pred_cm, clean=clean_cm, label=label_c, name=f"{name}_comp"
            # )

            # if (
            #     self.current_epoch == 0
            #     and self.trainer.local_rank == 0
            #     and batch_idx == 0
            # ):
            #     self.save_mrc(save_dir / "pred_cm.mrc", pred_cm.squeeze())
            #     self.save_mrc(save_dir / "clean_cm.mrc", clean_cm.squeeze())
            #     self.save_mrc(save_dir / "label_c.mrc", label_c.squeeze())

    @staticmethod
    def _cal_metrics(
        pred: torch.Tensor,  # type: ignore[valid-type]
        clean: Optional[torch.Tensor] = None,  # type: ignore[valid-type]
        label: Optional[torch.Tensor] = None,  # type: ignore[valid-type]
        iterations: Optional[int] = None,
        reduction: str = "mean_batch",
    ) -> Dict:
        """Calculates various image quality metrics between predicted and reference images."""
        scale_func = partial(DataHandler.scale_tensor, scale=True)

        metrics_dict = {}
        if clean is not None:
            for metric in ["LFD", "PSNR", "SSIM", "CC"]:
                metrics_dict[metric] = getattr(MetricsCalculator, metric)(
                    scale_func(clean),
                    scale_func(pred),
                    reduction=reduction,
                )
        if label is not None:
            metrics_dict["SNR"] = MetricsCalculator.SNR(
                noisys=scale_func(pred),
                masks=label,
                iterations=iterations,
                reduction=reduction,
            )

        return metrics_dict

    def _save_metrics(
        self,
        pred: torch.Tensor,  # type: ignore[valid-type]
        clean: Optional[torch.Tensor] = None,  # type: ignore[valid-type]
        label: Optional[torch.Tensor] = None,  # type: ignore[valid-type]
        name: str = "model_9",
        dataset: str = "SHREC",
        method: str = "Ours",
    ) -> None:
        """Saves performance metrics to a CSV file."""

        def get_format_dict(input_dict):
            output_dict = {}
            for k, v in input_dict.items():
                if isinstance(v, (int, float)):
                    before_len = len(str(int(abs(v))))
                    v = f"{v:.3f}" if before_len == 1 else f"{v:.2f}"
                output_dict[k] = str(v)
            return output_dict

        csv_path = (
            self.cfg_basic.infer_save_dir.parent / f"{self.cfg_basic.infer_save_dir.name}.csv"
        )

        metric_dict = {"Dataset": dataset, "Name": name, "Method": method}
        metrics = get_format_dict(
            self._cal_metrics(pred, clean=clean, label=label, iterations=self.iterations)
        )
        metric_dict.update(metrics)

        while True:
            if csv_path.exists():
                with open(csv_path, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(list(metric_dict.values()))
                break
            else:
                if self.trainer.local_rank == 0:
                    df = pd.DataFrame.from_dict(metric_dict, orient="index").T
                    df.to_csv(csv_path, index=False)
                    break
                else:
                    time.sleep(0.1)


def run_experiment(
    cfg_basic: DictConfig,
    cfg_model: DictConfig,
    name: str,
    ckpt_path: Optional[Union[str, Path]] = None,
) -> pl.LightningModule:
    """Execute training or inference based on the specified experiment mode.

    Args:
        cfg_basic: Basic configuration object containing global settings.
        cfg_model: Model configuration object defining architecture and hyperparameters.
        name: Experiment mode, e.g., "train" or "infer".
        ckpt_path: Optional checkpoint path.

    Returns:
        None
    """
    rank_zero_info(f"Launching experiment: {name}")

    # Determine the checkpoint path used for training or inference
    ckpt_path = ckpt_path or cfg_basic.pretrained_path
    rank_zero_info(f"Using checkpoint: {ckpt_path}")

    lit_net = LitNet(cfg_model, cfg_basic)

    # Prepare keyword arguments
    run_kwargs = {
        "model": lit_net,
        "datamodule": LitDataModule(get_transforms(cfg_basic), cfg_basic),
        "ckpt_path": ckpt_path,
    }

    # Instantiate trainer according to the experiment mode and configuration.
    trainer = get_trainer(name, cfg_basic)

    # Select the appropriate trainer method based on the experiment mode
    run_func = trainer.test if name == "infer" else trainer.fit
    run_func(**run_kwargs)

    return lit_net
