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
"""Step 2 (Optional)."""
from __future__ import annotations

import logging
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Union

import hyperopt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from cryojdm.nets import run_experiment
from cryojdm.utils import ConfigHandler, FileManager

warnings.filterwarnings("ignore")

seed_value = 42


cfg_search: Dict = {
    "max_evals": 500,  # 2,
    "batch_size": 2,
    "max_epochs": 100,  # 2,
    "check_val_every_n_epoch": 100,  # 2,
}

search_maps: Dict = {
    "ours": {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        "depths": [[2, 2, 2, 2], [2, 2, 3, 3], [3, 3, 3, 3]],
        "feature_size": [12, 24, 36, 48],
    },
    "emnet": {
        "warmup_epochs": [30, 40, 50],
        "lr": [5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        "momentum": [0.80, 0.85, 0.90, 0.95, 0.99],
        "nesterov": [True, False],
        "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        # "depths": [[3, 3, 3, 3]],
        # "conv_decoder": [False],
        # "fft_nums": [[2, 2, 0, 0], [0, 0, 2, 2], [0, 0, 0, 0], [2, 2, 2, 2]],
    },
    "umamba": {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    },
    "segresnet": {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        "dropout_prob": [0, 0.5, 0.1],
    },
    "unet": {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        "dropout": [0, 0.5, 0.1],
        "channels": [[16, 32, 64, 128, 256], [4, 8, 16, 32, 64]],
        "num_res_units": [2, 3, 4],
    },
    "unetr": {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        "feature_size": [16, 32],
        "dropout_rate": [0, 0.5, 0.1],
    },
    "swinunetr": {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        "depths": [[2, 2, 2, 2], [3, 3, 3, 3]],
        "num_heads": [[3, 6, 12, 24]],  # , [6, 12, 24, 48]
        "feature_size": [12, 24, 36, 48],
        "drop_rate": [0, 0.5, 0.1],
    },
}


def ours_spaces(
    cfg_model: DictConfig,
    search_choice: DictConfig,
) -> Dict:
    """Constructs a dictionary of hyperparameter spaces for optimization."""
    spaces = {
        "optim": {
            "lr": hyperopt.hp.choice("lr", search_choice.lr),
        },
        "net": {
            "depths": hyperopt.hp.choice("depths", search_choice.depths),
            "feature_size": hyperopt.hp.choice("feature_size", search_choice.feature_size),
        },
    }

    [(spaces.update({k: v}) if k not in spaces else None) for k, v in cfg_model.items()]  # type: ignore[dict-item]
    [
        (spaces["net"].update({k: v}) if k not in spaces["net"] else None)
        for k, v in cfg_model["net"].items()
    ]
    [
        (spaces["optim"].update({k: v}) if k not in spaces["optim"] else None)
        for k, v in cfg_model["optim"].items()
    ]
    return spaces


def emnet_spaces(
    cfg_model: DictConfig,
    search_choice: DictConfig,
) -> Dict:
    """Constructs a dictionary of hyperparameter spaces for optimization."""
    spaces = {
        "sched": {
            "warmup_epochs": hyperopt.hp.choice("warmup_epochs", search_choice.warmup_epochs),
        },
        "optim": {
            "lr": hyperopt.hp.choice("lr", search_choice.lr),
            "momentum": hyperopt.hp.choice("momentum", search_choice.momentum),
            "nesterov": hyperopt.hp.choice("nesterov", search_choice.nesterov),
            "weight_decay": hyperopt.hp.choice("weight_decay", search_choice.weight_decay),
        },
        # "net": {
        #     "depths": hyperopt.hp.choice("depths", search_choice.depths),
        #     "conv_decoder": hyperopt.hp.choice(
        #         "conv_decoder", search_choice.conv_decoder
        #     ),
        #     "fft_nums": hyperopt.hp.choice("fft_nums", search_choice.fft_nums),
        # },
    }

    [(spaces.update({k: v}) if k not in spaces else None) for k, v in cfg_model.items()]  # type: ignore[dict-item]
    [
        (spaces["sched"].update({k: v}) if k not in spaces["sched"] else None)
        for k, v in cfg_model["sched"].items()
    ]
    [
        (spaces["optim"].update({k: v}) if k not in spaces["optim"] else None)
        for k, v in cfg_model["optim"].items()
    ]
    # [
    #     (spaces["net"].update({k: v}) if k not in spaces["net"] else None)
    #     for k, v in cfg_model["net"].items()
    # ]
    return spaces


def umamba_spaces(
    cfg_model: DictConfig,
    search_choice: DictConfig,
) -> Dict:
    """Constructs a dictionary of hyperparameter spaces for optimization."""
    spaces = {
        "optim": {
            "lr": hyperopt.hp.choice("lr", search_choice.lr),
        },
    }

    [(spaces.update({k: v}) if k not in spaces else None) for k, v in cfg_model.items()]  # type: ignore[dict-item]
    [
        (spaces["net"].update({k: v}) if k not in spaces["net"] else None)
        for k, v in cfg_model["net"].items()
    ]
    [
        (spaces["optim"].update({k: v}) if k not in spaces["optim"] else None)
        for k, v in cfg_model["optim"].items()
    ]
    return spaces


def segresnet_spaces(
    cfg_model: DictConfig,
    search_choice: DictConfig,
) -> Dict:
    """Constructs a dictionary of hyperparameter spaces for optimization."""
    drop_rate_samples = [round(float(d), 1) for d in np.arange(*search_choice.dropout_prob)]

    spaces = {
        "optim": {
            "lr": hyperopt.hp.choice("lr", search_choice.lr),
        },
        "net": {
            "dropout_prob": hyperopt.hp.choice("dropout_prob", drop_rate_samples),
        },
    }

    [(spaces.update({k: v}) if k not in spaces else None) for k, v in cfg_model.items()]  # type: ignore[dict-item]
    [
        (spaces["net"].update({k: v}) if k not in spaces["net"] else None)
        for k, v in cfg_model["net"].items()
    ]
    [
        (spaces["optim"].update({k: v}) if k not in spaces["optim"] else None)
        for k, v in cfg_model["optim"].items()
    ]
    return spaces


def unet_spaces(
    cfg_model: DictConfig,
    search_choice: DictConfig,
) -> Dict:
    """Constructs a dictionary of hyperparameter spaces for optimization."""
    drop_rate_samples = [round(float(d), 1) for d in np.arange(*search_choice.dropout)]

    spaces = {
        "optim": {
            "lr": hyperopt.hp.choice("lr", search_choice.lr),
        },
        "net": {
            "channels": hyperopt.hp.choice("channels", search_choice.channels),
            "num_res_units": hyperopt.hp.choice("num_res_units", search_choice.num_res_units),
            "dropout": hyperopt.hp.choice("dropout", drop_rate_samples),
        },
    }

    [(spaces.update({k: v}) if k not in spaces else None) for k, v in cfg_model.items()]  # type: ignore[dict-item]
    [
        (spaces["net"].update({k: v}) if k not in spaces["net"] else None)
        for k, v in cfg_model["net"].items()
    ]
    [
        (spaces["optim"].update({k: v}) if k not in spaces["optim"] else None)
        for k, v in cfg_model["optim"].items()
    ]
    return spaces


def unetr_spaces(
    cfg_model: DictConfig,
    search_choice: DictConfig,
) -> Dict:
    """Constructs a dictionary of hyperparameter spaces for optimization."""
    drop_rate_samples = [round(float(d), 1) for d in np.arange(*search_choice.dropout_rate)]

    spaces = {
        "optim": {
            "lr": hyperopt.hp.choice("lr", search_choice.lr),
        },
        "net": {
            "feature_size": hyperopt.hp.choice("feature_size", search_choice.feature_size),
            "dropout_rate": hyperopt.hp.choice("dropout_rate", drop_rate_samples),
        },
    }

    [(spaces.update({k: v}) if k not in spaces else None) for k, v in cfg_model.items()]  # type: ignore[dict-item]
    [
        (spaces["net"].update({k: v}) if k not in spaces["net"] else None)
        for k, v in cfg_model["net"].items()
    ]
    [
        (spaces["optim"].update({k: v}) if k not in spaces["optim"] else None)
        for k, v in cfg_model["optim"].items()
    ]
    return spaces


def swinunetr_spaces(
    cfg_model: DictConfig,
    search_choice: DictConfig,
) -> Dict:
    """Constructs a dictionary of hyperparameter spaces for optimization."""
    drop_rate_samples = [round(float(d), 1) for d in np.arange(*search_choice.drop_rate)]

    spaces = {
        "optim": {
            "lr": hyperopt.hp.choice("lr", search_choice.lr),
        },
        "net": {
            "depths": hyperopt.hp.choice("depths", search_choice.depths),
            "num_heads": hyperopt.hp.choice("num_heads", search_choice.num_heads),
            "feature_size": hyperopt.hp.choice("feature_size", search_choice.feature_size),
            "drop_rate": hyperopt.hp.choice("drop_rate", drop_rate_samples),
            "attn_drop_rate": hyperopt.hp.choice("attn_drop_rate", drop_rate_samples),
            "dropout_path_rate": hyperopt.hp.choice("dropout_path_rate", drop_rate_samples),
        },
    }

    [(spaces.update({k: v}) if k not in spaces else None) for k, v in cfg_model.items()]  # type: ignore[dict-item]
    [
        (spaces["net"].update({k: v}) if k not in spaces["net"] else None)
        for k, v in cfg_model["net"].items()
    ]
    [
        (spaces["optim"].update({k: v}) if k not in spaces["optim"] else None)
        for k, v in cfg_model["optim"].items()
    ]
    return spaces


spaces_maps: Dict = {
    "ours": ours_spaces,
    "emnet": emnet_spaces,
    "umamba": umamba_spaces,
    "segresnet": segresnet_spaces,
    "unet": unet_spaces,
    "unetr": unetr_spaces,
    "swinunetr": swinunetr_spaces,
}


def run_search(
    cfgs: DictConfig,
    cfg_search: Dict = cfg_search,
) -> DictConfig:
    """Executes the searching process using the provided configuration."""

    def _main_rank_search(params: Dict, cfgs: DictConfig) -> Dict:
        """Executes the main rank search process."""
        cfgs[cfgs.tmp.model] = params
        info = f"Update search arguments: {params}"
        logging.info(info)

        search_dir = cfgs.tmp.search_dir
        yamls = sorted(search_dir.glob("version_*.yaml"))
        latest_num = int(yamls[-1].stem.split("_")[-1]) + 1 if yamls else 0

        ConfigHandler.save(cfgs, f"{search_dir}/version_{latest_num:03d}.yaml")
        lit_net = run_experiment(cfg_basic=cfgs.tmp, cfg_model=cfgs[cfgs.tmp.model], name="search")
        ConfigHandler.save(cfgs, f"{search_dir}/version_{latest_num:03d}/trained.yaml")
        return {"loss": lit_net.best_val_loss, "status": "ok"}

    def _other_ranks_search(cfg_tmp: DictConfig, load_path: Union[str, Path]) -> None:
        """Executes the other ranks search process."""
        while True:
            try:
                cfgs = OmegaConf.load(load_path)
                break
            except FileNotFoundError:
                time.sleep(0.1)

        _ = run_experiment(cfg_basic=cfg_tmp, cfg_model=cfgs[cfg_tmp.model], name="search")

    cfg_tmp = cfgs.tmp
    cfg_model = cfgs[cfgs.tmp.model]
    cfg_search = OmegaConf.create(cfg_search)  # type: ignore[assignment]
    search_choice = OmegaConf.create(search_maps[cfgs.tmp.model])  # type: ignore[call-overload]

    first_yaml = cfg_tmp.search_dir / "version_000.yaml"
    best_yaml = cfg_tmp.search_dir / "best.yaml"

    for key in ["batch_size", "max_epochs", "check_val_every_n_epoch"]:
        cfg_tmp[key] = cfg_search[key]
    cfg_tmp.sw_batch_size = cfg_tmp.batch_size

    if first_yaml.exists():
        for t in range(cfg_search.max_evals):  # type: ignore[attr-defined]
            search_yaml = cfg_tmp.search_dir / f"version_{t:03d}.yaml"
            _other_ranks_search(cfg_tmp, search_yaml)
    else:
        FileManager.update(cfg_tmp.search_dir)
        info = f"Default arguments: {cfg_model}"
        logging.info(info)

        spaces = spaces_maps[cfg_tmp.model](cfg_model, search_choice)
        main_rank_search = partial(_main_rank_search, cfgs=cfgs)
        trials = hyperopt.Trials()
        best = hyperopt.fmin(
            fn=main_rank_search,
            space=spaces,
            algo=hyperopt.tpe.suggest,
            max_evals=cfg_search.max_evals,  # type: ignore[attr-defined]
            trials=trials,
            rstate=np.random.default_rng(seed_value),
            show_progressbar=True,
        )

        best_params = hyperopt.space_eval(spaces, best)
        info = f"Best params: {best_params}"
        logging.info(info)
        cfgs[cfg_tmp.model] = best_params
        ConfigHandler.save(cfgs, best_yaml)

        losses = [x["result"]["loss"] for x in trials.trials]
        search_df = pd.DataFrame(
            {
                "step": range(len(losses)),
                "loss": losses,
                "min_loss": [np.min(losses[: i + 1]) for i in range(len(losses))],
            }
        )
        search_df.to_csv(
            cfg_tmp.search_dir / "loss.csv",
            index=False,
            header=True,
            float_format="%.5f",
        )

    while True:
        try:
            cfgs_best = OmegaConf.load(best_yaml)
            break
        except FileNotFoundError:
            time.sleep(0.1)

    cfgs[cfg_tmp.model] = cfgs_best[cfg_tmp.model]

    logging.info("Training with best arguments...")
    return cfgs
