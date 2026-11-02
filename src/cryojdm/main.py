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
"""Main pipeline (search-train-infer)."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Union

from monai.utils import set_determinism
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info

from cryojdm.nets import get_latest_version, run_experiment
from cryojdm.search import run_search
from cryojdm.utils import ConfigHandler, FileHandler, path_resolve

warnings.filterwarnings("ignore")
load_json = FileHandler.load_json


import torch

torch.set_float32_matmul_precision("high")


def create_parser() -> argparse.ArgumentParser:
    """Set up an argument parser for command-line input.

    Returns:
        argparse.ArgumentParser: Configured parser with expected arguments.
    """
    parser = argparse.ArgumentParser(
        description="Argument parser for configuring training and inference."
    )

    # Required argument group
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file defining the training or inference setup.",
    )

    # Basic config group
    config_group = parser.add_argument_group("Basic config parameters")
    config_group.add_argument(
        "--base_dir",
        type=str,
        help="Root directory for all input and output files.",
    )
    config_group.add_argument(
        "--json_path",
        type=str,
        help="Path to the JSON file containing dataset information.",
    )

    config_group.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    config_group.add_argument(
        "--roi_len",
        type=int,
        help="Length of the Region Of Interest (ROI) for subvolume extraction.",
    )
    config_group.add_argument(
        "--model",
        type=str,
        help="Name of the model architecture for training or inference.",
    )
    config_group.add_argument(
        "--devices",
        type=int,
        nargs="+",
        help="List of GPU device IDs (space-separated) or 'cpu'.",
    )
    config_group.add_argument(
        "--memo",
        type=str,
        help="Optional description for this experiment.",
    )

    # Training parameters group
    train_group = parser.add_argument_group("Training parameters")
    train_group.add_argument(
        "--lr",
        type=float,
        help="Learning rate for training.",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training.",
    )
    train_group.add_argument(
        "--max_epochs",
        type=int,
        help="Maximum number of epochs for training.",
    )
    train_group.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        help="Frequency (in epochs) to run validation during training.",
    )
    train_group.add_argument(
        "--num_workers",
        type=int,
        help="Number of parallel workers for loading training data.",
    )

    train_group.add_argument(
        "--pretrained_path",
        type=str,
        help="Path to the pretrained model weights.",
    )
    train_group.add_argument(
        "--mode",
        type=int,
        help="Operational mode. 0 for ours, any other value for Noise2Noise training.",
    )

    # Inference parameters group
    infer_group = parser.add_argument_group("Inference parameters")
    infer_group.add_argument(
        "--infer",
        type=str,
        help="Specifies the path or version of the checkpoint to use for inference.",
    )
    infer_group.add_argument(
        "--sw_batch_size",
        type=int,
        help="Sliding window batch size used during inference.",
    )
    infer_group.add_argument(
        "--infer_overlap",
        type=float,
        help="Overlap ratio (0.0 to 1.0) for subvolumes during inference.",
    )

    return parser


class ConfigCustom:
    """Provide flexible operations on config object.

    Offers features including loading and updating.
    """

    def __init__(self) -> None:
        """Initialize by parsing command line args and loading the default config."""
        self.args_ori = create_parser().parse_args()
        self.cfgs_ori = OmegaConf.load(path_resolve(self.args_ori.config_file))

    def update(self, cfgs: DictConfig) -> DictConfig:
        """Update config with command-line arguments and computed defaults."""

        def _set_key(cfgs: DictConfig, key: str, value: Optional[Union[str, int]] = None) -> None:
            """Set a default value to the key if it doesn't exist or its value is empty."""
            cfgs[key] = cfgs.get(key) or value

        def _update_basic_cfg(
            args: argparse.ArgumentParser, cfg_basic: DictConfig, cfg_model: DictConfig
        ) -> None:
            """Update basic config with command-line args."""
            base_dir = path_resolve(cfg_basic.base_dir)

            cfg_basic.update(
                {k: v for k, v in vars(args).items() if v is not None and k in cfg_basic}
            )

            _set_key(cfg_basic, "infer", None)
            _set_key(cfg_basic, "memo", " ")
            _set_key(cfg_basic, "json_path", str(base_dir / "1_preprocess/json/train.json"))

            json_data = load_json(cfg_basic.json_path)
            _, voxel_size = FileHandler.load_mrc(json_data["train"][0]["all"], get_voxel_size=True)

            dataset_len = len(json_data["train"])
            basic_updates = {
                "dataset_len": dataset_len,
                "infer_mode": len(json_data["infer"][0]),
                "voxel_size": float(max(voxel_size)),
                "batch_size": min(
                    cfg_basic.get("batch_size"),
                    cfg_model.get("max_batch_size", float("inf")),
                ),
            }
            cfg_basic.update(basic_updates)

        def _update_model_cfg(
            args: argparse.ArgumentParser, cfg_basic: DictConfig, cfg_model: DictConfig
        ) -> None:
            """Update model config based on basic config and command-line args."""
            model_updates = {
                "img_size": cfg_basic["roi_len"],
                "spatial_dims": cfg_basic["spatial_dims"],
                "in_shape": [int(cfg_basic["roi_len"])] * cfg_basic["spatial_dims"],
            }

            for k, v in model_updates.items():
                if k in cfg_model["net"]:
                    cfg_model["net"][k] = v

            if args.lr is not None:
                cfg_model["optim"]["lr"] = args.lr

            cfg_model["net"] = OmegaConf.merge(
                (cfg_model.get("net") or OmegaConf.create()),
                (cfg_ablation.get("net") or OmegaConf.create()),
            )

        def _update_tmp_cfg(
            cfgs: DictConfig, cfg_basic: DictConfig, cfg_ablation: DictConfig
        ) -> None:
            """Add temporary config settings."""
            base_dir = path_resolve(cfg_basic.base_dir)

            cfgs.tmp = OmegaConf.merge(cfg_basic, cfg_ablation.basic)
            cfgs.tmp.output_dir = (
                base_dir / "2_train" / f"{cfgs.tmp.model}_seed_{cfgs.tmp.seed:04d}"
            )
            cfgs.tmp.train_dir = cfgs.tmp.output_dir / "train"
            cfgs.tmp.search_dir = cfgs.tmp.output_dir / "search"
            cfgs.tmp.train_dir.mkdir(exist_ok=True, parents=True)

        args = self.args_ori
        cfg_basic, cfg_model, cfg_ablation = (
            cfgs.basic,
            cfgs[cfgs.basic.model],
            cfgs.ablation,
        )

        _update_basic_cfg(args, cfg_basic, cfg_model)
        _update_model_cfg(args, cfg_basic, cfg_model)
        _update_tmp_cfg(cfgs, cfg_basic, cfg_ablation)

        return cfgs

    def _load_and_update_cfgs(self, trained_dir: Path, training_dir: Path) -> DictConfig:
        """Load and update config from the specified 'trained.yaml' or 'training.yaml' file."""
        trained_yaml = trained_dir / "trained.yaml"
        training_yaml = training_dir / "training.yaml"
        file_path = trained_yaml if trained_yaml.is_file() else training_yaml
        cfgs = OmegaConf.load(file_path)
        return self.update(cfgs)

    def get_pretrained(self, cfgs: DictConfig) -> DictConfig:
        """Load and update config from a specified pretrained model for training."""
        pretrained_path = cfgs.tmp.pretrained_path
        trained_dir = Path(pretrained_path.split("checkpoints")[0])
        training_dir = trained_dir.parent

        cfgs_new = self._load_and_update_cfgs(trained_dir, training_dir)
        return cfgs_new

    def get_infer(self, cfgs: DictConfig) -> DictConfig:
        """Load and update config from a specified trained model for inference."""
        infer_input = cfgs.tmp.infer
        training_dir = Path(cfgs.tmp.train_dir)

        if ".ckpt" in str(infer_input):
            infer_ckpt = infer_input
            infer_num = int(infer_input.split("version_")[-1].split("/")[0])
        else:
            infer_ckpt = None
            infer_num = int(infer_input)

        trained_dir = training_dir / f"version_{infer_num:03d}"

        cfgs_new = self._load_and_update_cfgs(trained_dir, training_dir)
        cfgs_new.tmp.infer_ckpt = infer_ckpt
        cfgs_new.tmp.infer = infer_num
        return cfgs_new


def run_train(cfgs: DictConfig) -> DictConfig:
    """Execute the training process using the provided config.

    Args:
        cfgs (DictConfig): The config object containing all relevant settings.

    Returns:
        DictConfig: The updated config object after training.
    """
    # Save the initial config
    train_dir = cfgs.tmp.train_dir
    ConfigHandler.save(cfgs, train_dir / "training.yaml")

    # Start the training process
    _ = run_experiment(cfg_basic=cfgs.tmp, cfg_model=getattr(cfgs, cfgs.tmp.model), name="train")

    # Determine the latest training version & Save the updated config
    latest_version = get_latest_version(train_dir)
    cfgs.tmp.infer = cfgs.basic.infer = latest_version
    save_dir = train_dir / f"version_{latest_version:03d}"
    ConfigHandler.save(cfgs, save_dir / "trained.yaml")

    # Save the training memo
    (save_dir / "memo.txt").write_text(f"\n{save_dir.stem}: {cfgs.basic.memo}\n")

    rank_zero_info("Finish training!")
    return cfgs


def run_infer(cfgs: DictConfig) -> DictConfig:
    """Execute the inference process using the specified configs.

    Args:
        cfgs (DictConfig): The config object containing all relevant settings.

    Returns:
        DictConfig: The updated config object after inference.
    """
    checkpoint = cfgs.tmp.get("infer_ckpt")
    ckpt_dir = cfgs.tmp.train_dir / f"version_{cfgs.tmp.infer:03d}" / "checkpoints/top_k"  # top_k

    def _run_infer(checkpoint_path: Path) -> None:
        """Run inference using a specific checkpoint."""
        cfgs.tmp.infer_save_dir = ckpt_dir / f"infer_{checkpoint_path.stem}"
        cfgs.tmp.infer_save_dir.mkdir(exist_ok=True, parents=True)
        info = f"Start inference with checkpoint: {checkpoint_path} ..."
        rank_zero_info(info)
        _ = run_experiment(
            cfg_basic=cfgs.tmp,
            cfg_model=getattr(cfgs, cfgs.tmp.model),
            ckpt_path=checkpoint_path,
            name="infer",
        )

    if checkpoint:
        _run_infer(path_resolve(checkpoint))
    else:
        for checkpoint in sorted(ckpt_dir.glob("*.ckpt"), reverse=True):
            _run_infer(checkpoint)

    rank_zero_info("Finish inference!")
    return cfgs


def launch_pipeline() -> None:
    """Initialize and run the pipeline."""
    # Step 1: Basic config setup
    cfc = ConfigCustom()
    cfgs = cfc.update(cfc.cfgs_ori)  # type: ignore[arg-type]

    # Set global random seed for reproducibility
    seed = cfgs.tmp.seed
    set_determinism(seed)
    seed_everything(seed, workers=True)

    # Step 2: Execute the chosen process
    if "search" in cfgs.tmp.memo:
        # Step 2.1: Execute searching process (optional)
        rank_zero_info("Start searching...")
        cfgs = run_search(cfgs)

    if cfgs.tmp.infer is None:
        # Step 2.2: Execute training process (optional)
        if cfgs.tmp.pretrained_path:
            cfgs = cfc.get_pretrained(cfgs)
            info = f"Resume training from: {cfgs.tmp.pretrained_path}..."
        else:
            info = "Start training from scratch..."
        rank_zero_info(info)
        cfgs = run_train(cfgs)
        if "infer" in cfgs.tmp.memo:
            cfgs = run_infer(cfgs)
    else:
        # Step 2.3: Execute inference process (optional)
        cfgs = cfc.get_infer(cfgs)
        cfgs = run_infer(cfgs)


if __name__ == "__main__":
    launch_pipeline()
