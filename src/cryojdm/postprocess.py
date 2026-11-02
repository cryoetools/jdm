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
"""Post-process."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from omegaconf import OmegaConf

from cryojdm.utils import DataHandler, FileHandler, path_resolve

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="\n%(asctime)s, %(levelname)-8s %(message)s\n",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

load_mrc = FileHandler.load_mrc
save_mrc = FileHandler.save_mrc
norm_func = partial(DataHandler.scale_array, norm=True)


def create_parser() -> argparse.ArgumentParser:
    """Set up an argument parser for command-line input.

    Returns:
        argparse.ArgumentParser: Configured parser with expected arguments.
    """
    parser = argparse.ArgumentParser(description="Process command-line arguments.")
    parser.add_argument(
        "--config_file",
        type=str,
        required=False,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--base_dir", type=str, help="Root directory for all input and output files."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model architecture for training or inference.",
    )

    return parser


class MRCPredMean:
    """"""

    def __init__(
        self,
        base_dir: Union[Path, str],
        model: Optional[str] = "ours",
        seed_lst: Optional[List] = [3115, 58, 42],  # TODO
    ):
        """"""
        self.base_dir = path_resolve(base_dir)
        self.seed_dirs = [
            self.base_dir / "2_train" / f"{model}_seed_{i:04d}" / "train"  # "train"## TODO
            for i in seed_lst
        ]

    def _get_mrc_path(self, seed_dir: Path, version_name: str, file_name: str) -> Path | None:
        """"""
        search_path = seed_dir / version_name / "checkpoints" / "top_k"
        matched = list(search_path.rglob(file_name))
        return sorted(matched)[-1] if matched else None

    def _compute_mean_prediction(self, mrc_paths: List[Path]) -> np.ndarray:
        """"""
        predictions = [load_mrc(p) for p in mrc_paths]
        predictions = norm_func(np.mean(np.stack(predictions), axis=0))

        _, voxel_size = FileHandler.load_mrc(mrc_paths[0], get_voxel_size=True)
        self.voxel_size = float(max(voxel_size))
        return predictions

    def run(self):
        """"""
        version_dirs = [
            d
            for d in sorted(self.seed_dirs[0].glob("version_*"))
            if not (d / "pred_mean").exists()
        ]

        for version_dir in version_dirs:
            version_name = version_dir.name
            top_k_dir = sorted((version_dir / "checkpoints" / "top_k").rglob("*.mrc"))[-1].parent
            mrc_files = sorted(top_k_dir.glob("*.mrc"))

            for mrc_file in mrc_files:
                mrc_name = mrc_file.name
                mrc_paths = [
                    self._get_mrc_path(seed_dir, version_name, mrc_name)
                    for seed_dir in self.seed_dirs
                ]

                if all(mrc_paths):
                    mean_pred = self._compute_mean_prediction(mrc_paths)
                    save_dir = version_dir / "pred_mean"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    save_path = save_dir / mrc_name
                    save_mrc(save_path, mean_pred, voxel_size=self.voxel_size)
                    logging.info(f"Save {save_path}")

                    # (save_dir / "log.txt").write_text(f"Save:\n{mrc_paths}\n")


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.config_file is not None:
        cfgs = OmegaConf.load(args.config_file)
        base_dir = cfgs.basic.base_dir
        model = cfgs.basic.model

    if args.base_dir:
        base_dir = args.base_dir

    if args.model:
        model = args.model

    averager = MRCPredMean(base_dir=Path(base_dir), model=model)
    averager.run()
