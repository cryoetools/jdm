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
"""Module utils."""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
import starfile
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from matplotlib.colors import ListedColormap
from monai import transforms
from monai.config import print_config
from monai.metrics import MultiScaleSSIMMetric, PSNRMetric, SSIMMetric
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.spatial.transform import Rotation as R  # noqa: N817
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="\n%(asctime)s, %(levelname)-8s %(message)s\n",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
path_resolve = lambda p: Path(p).resolve()  # noqa: E731


class DataHandler:
    """Provides flexible operations on arrays (NumPy) and tensor batches (PyTorch).

    Offers features including scaling, normalization, resizing,
    rotation, downsampling, cropping, etc.

    Example:
        .. code-block:: python

            arr = np.random.rand(128, 128, 128)
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            dh = DataHandler()

            # Clipping example
            clipped = dh.scale_array(arr, clip=True, clip_lower=10, clip_upper=90)
            print(arr.min(), arr.max(), clipped.min(), clipped.max())

            # Scaling by a factor
            scaled = dh.scale_array(arr, scale=True, scale_factor=2.0)
            print(arr.min(), arr.max(), scaled.min(), scaled.max())

            # Scaling to a custom range
            scaled = dh.scale_array(arr, scale=True, scale_min=0, scale_max=2)
            print(arr.min(), arr.max(), scaled.min(), scaled.max())

            # Scaling a tensor batch
            scaled = dh.scale_tensor(x, scale=True)
            print(x.min(), x.max(), scaled.min(), scaled.max())

            # Normalizing a tensor batch
            normalized = dh.scale_tensor(x, norm=True)
            print(x.mean(), x.std(), normalized.mean(), normalized.std())

            # Resizing a PyTorch tensor
            resized = dh.resize_tensor(x, target_shape=(64, 64, 64))
            print(x.shape, resized.shape)

            # Rotating a 3D tensor batch
            rotated = dh.rotate_tensor(x, axis="z", angle=45)
            print(rotated.shape)

            # Downsample a 3D tensor batch
            out1, out2 = dh.downsample_tensor(x)
            print(out1.shape, out2.shape)

            # Cropping the center of an array
            cropped = dh.crop_center(arr, target_shape=(64, 64, 64))
            print(arr.shape, cropped.shape)

            # Slicing the center of an array
            sliced = dh.slice_center(arr, thickness=100, axis=-1)
            print(arr.shape, sliced.shape)
    """

    @staticmethod
    def scale_array(
        arr: np.ndarray,
        clip: bool = False,
        clip_lower: float = 5,
        clip_upper: float = 95,
        scale: float = False,
        scale_min: Optional[float] = None,
        scale_max: Optional[float] = None,
        scale_factor: float = 1.0,
        norm: bool = False,
        output_type: type = np.float32,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """Applies clipping, scaling, or normalization to arrays.

        Args:
            arr (np.ndarray):
                Input arrays.
            clip (bool):
                If True, clips the array to specified percentiles (default=False).
            clip_lower (float):
                Lower percentile for clipping. Ignored if `clip` is False (default=5).
            clip_upper (float):
                Upper percentile for clipping. Ignored if `clip` is False (default=95).
            scale (bool):
                If True, scales the array to a specified range (default=False).
            scale_min (Optional[float]):
                Minimum value for scaling. Ignored if `scale` is False (default=None).
            scale_max (Optional[float]):
                Maximum value for scaling. Ignored if `scale` is False (default=None).
            scale_factor (float):
                Factor to scale the array by (default=1.0).
            norm (bool):
                If True, normalizes to zero mean and unit variance (default=False).
            output_type (type):
                Desired output data type of the array (default=np.float32).
            epsilon (float):
                Small value to prevent division by zero (default=1e-6).

        Returns:
            np.ndarray: Processed array.
        """
        arr = np.asarray(arr)

        # Percentile clipping
        if clip:
            lower_val = np.percentile(arr, clip_lower)
            upper_val = np.percentile(arr, clip_upper)
            arr = np.clip(arr, lower_val, upper_val)

        # Scale or Normalize
        if scale:
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            range_val = np.clip(arr_max - arr_min, epsilon, None)

            if scale_min is not None and scale_max is not None:
                arr = (arr - arr_min) / range_val * (scale_max - scale_min) + scale_min
            else:
                arr = (arr - arr_min) / range_val * scale_factor

        elif norm:
            mean_val = arr.mean()
            std_val = np.clip(arr.std(), epsilon, None)
            arr = (arr - mean_val) / std_val

        return arr.astype(output_type)

    @staticmethod
    def scale_tensor(
        x: torch.Tensor,
        scale: bool = False,
        norm: bool = False,
        epsilon: float = 1e-6,
        dims: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Apply scaling or normalization to a tensor.

        This function performs per-sample scaling to [0, 1] or normalization
        to zero mean and unit variance. By default, the scaling/normalization
        is applied across all non-batch dimensions.

        Args:
            x (torch.Tensor):
                Input tensor. The first dimension is treated as batch.
            scale (bool):
                Whether to apply min-max scaling to [0, 1].
            norm (bool):
                Whether to apply zero-mean unit-variance normalization.
            epsilon (float):
                Small constant to avoid division by zero.
            dims (Sequence[int], optional):
                Dimensions over which to compute statistics. If None,
                all dimensions except the batch dimension are used.

        Returns:
            torch.Tensor: The processed tensor.

        Raises:
            ValueError: If both `scale` and `norm` are True.
            ValueError: If `x` has fewer than 2 dimensions.
        """
        # Validate tensor dimension。
        if x.ndim < 2:
            raise ValueError("Input tensor must have at least 2 dimensions (batch + features).")
        if dims is None:
            dims = tuple(range(1, x.ndim))
        else:
            dims = tuple(dims)

        # min-max scaling。
        if scale:
            min_val = x.amin(dim=dims, keepdim=True)
            max_val = x.amax(dim=dims, keepdim=True)
            scale_range = (max_val - min_val).clamp_min(epsilon)
            return (x - min_val) / scale_range

        # normalization: zero mean + unit variance。
        if norm:
            mean = x.mean(dim=dims, keepdim=True)
            std = x.std(dim=dims, keepdim=True).clamp_min(epsilon)
            return (x - mean) / std
        return x

    @staticmethod
    def resize_tensor(
        x: torch.Tensor, target_shape: Tuple, mode: str = "trilinear"
    ) -> torch.Tensor:
        """Resizes tensor batches to the specified shape using interpolation.

        Args:
            x (torch.Tensor): Input tensor batches.
            target_shape (Tuple): Desired output shape.
            mode (str): Interpolation mode (default="trilinear").
            The modes available: `nearest`, `linear` (3D-only), `bilinear`,
            `bicubic` (4D-only), `trilinear` (5D-only), `area`, `nearest-exact`

        Returns:
            torch.Tensor: Processed tensor.
        """
        return F.interpolate(x, size=target_shape, mode=mode)  # type: ignore[arg-type]

    @staticmethod
    def rotate_tensor(
        x: torch.Tensor, axis: str = "z", angle: int = 90, mask_flag: bool = False
    ) -> torch.Tensor:
        """Rotates tensor batches (3D) around a specified axis by a given angle.

        Args:
            x (torch.Tensor):
                Input tensor batches with shape (B, C, D, H, W).
            axis (str):
                The axis to rotate around. One of "x", "y", or "z" (default="z").
            angle (int):
                The angle of rotation in degrees (default=90).
            mask_flag (bool):
                If True, uses nearest neighbor interpolation (useful for masks).
                If False, uses bilinear interpolation (default=False).

        Returns:
            torch.Tensor: Processed tensor.
        """
        size = x.size()
        affine = np.eye(4)
        affine[:3, :3] = R.from_euler(axis, np.deg2rad(angle)).as_matrix()
        affine = (
            torch.from_numpy(affine[:3, :])  # type: ignore[assignment]
            .unsqueeze(0)
            .expand(size[0], -1, -1)
            .to(torch.float32)
            .to(x.device)
        )
        grid = F.affine_grid(affine, size)  # type: ignore[arg-type]

        if mask_flag:
            func = partial(F.grid_sample, mode="nearest", padding_mode="zeros", grid=grid)
            return func(x.to(torch.float32)).to(torch.int8)
        else:
            func = partial(F.grid_sample, mode="bilinear", padding_mode="reflection", grid=grid)
            return func(x.to(torch.float32))

    @staticmethod
    def downsample_tensor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downsamples tensor batches (2D or 3D) by half using convolution operations.

        Args:
            x (torch.Tensor): Input tensor batches.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Two downsampled versions of the input tensor batches.

        Note:
            Adapted from:
            https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b
        """
        dim = x.dim() - 2
        c = x.size(1)
        base = torch.eye(2, device=x.device, dtype=x.dtype)

        if dim == 2:
            filter1 = base.unsqueeze(0).unsqueeze(0).expand(c, -1, -1, -1)
            conv_func = F.conv2d
        elif dim == 3:
            filter1 = (
                torch.stack([base, base.flip(1)])
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(c, -1, -1, -1, -1)
            )
            conv_func = F.conv3d
        else:
            raise ValueError("Unsupported dimension. Supported values are 2 or 3.")

        filter2 = 1 - filter1
        output1 = conv_func(x, filter1 * 0.5, stride=2, groups=c)
        output2 = conv_func(x, filter2 * 0.5, stride=2, groups=c)

        return output1, output2

    @staticmethod
    def crop_center(
        x: Union[np.ndarray, torch.Tensor], target_shape: Sequence[int]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Crop the center of arrays or tensor batches to a given target shape.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input arrays or tensor batches.
            target_shape (Sequence[int]): Desired output shape.

        Returns:
            Union[np.ndarray, torch.Tensor]: Cropped data.
        """
        dim = len(target_shape)
        input_shape = x.shape[-dim:]

        if all(i == t for i, t in zip(input_shape, target_shape, strict=False)):
            return x

        crop_slice = []
        for i in range(dim):
            start = (input_shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            crop_slice.append(slice(start, end))

        return x[(..., *tuple(crop_slice))]  # type: ignore[arg-type]

    @staticmethod
    def slice_center(
        x: Union[np.ndarray, torch.Tensor], thickness: int, axis: int = 0
    ) -> Union[np.ndarray, torch.Tensor]:
        """Slices the center of arrays or tensor batches with a given thickness.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input arrays or tensor batches.
            thickness (int): Target thickness along the specified axis.
            axis (int): The axis along which to slice (default=0).

        Returns:
            Union[np.ndarray, torch.Tensor]: Sliced data.
        """
        shape = x.shape

        start_idx = (shape[axis] - thickness) // 2  # type: ignore[index]
        end_idx = start_idx + thickness

        slices = [slice(None)] * len(shape)
        slices[axis] = slice(start_idx, end_idx)  # type: ignore[index]
        return x[tuple(slices)]


class FileHandler:
    """Handles JSON, YAML, and MRC file operations.

    Provides methods for loading and saving data in JSON, YAML, or MRC formats.

    Example:
        .. code-block:: python

            fh = FileHandler()

            # JSON Operations
            json_path = "example.json"
            fh.save_json(json_path, {"key": "value"})
            json_data = fh.load_json(json_path)
            print(json_data)

            # YAML Operations
            yaml_path = "example.yaml"
            fh.save_yaml(yaml_path, {"key": "value"})
            yaml_data = fh.load_yaml(yaml_path)
            print(yaml_data)

            # MRC Operations
            data = np.random.rand(10, 10, 10)
            fh.save_mrc("example.mrc", data, voxel_size=10.0)
            data, voxel_size = fh.load_mrc("example.mrc", get_voxel_size=True)
            print(data.shape, voxel_size)

    """

    @staticmethod
    def load_json(load_path: Union[str, Path]) -> Dict:
        """Loads data from a JSON file.

        Args:
            load_path (Union[str, Path]): Path to the JSON file.

        Returns:
            Dict: Data loaded from the JSON file.
        """
        with open(load_path, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def save_json(save_path: Union[str, Path], save_data: Dict) -> None:
        """Saves data to a JSON file.

        Args:
            save_path (Union[str, Path]): Path to the JSON file.
            save_data (Dict): Data to save.
        """
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=4)
        logging.info(f"Save {save_path}")  # noqa: G004

    @staticmethod
    def load_yaml(load_path: Union[str, Path]) -> Union[DictConfig, ListConfig]:
        """Loads data from a YAML file.

        Args:
            load_path (Union[str, Path]): Path to the YAML file.

        Returns:
            OmegaConf: Data loaded from the YAML file.
        """
        logging.info(f"Load {load_path}")  # noqa: G004
        return OmegaConf.load(load_path)

    @staticmethod
    def save_yaml(save_path: Union[str, Path], save_data: Dict) -> None:
        """Saves data to a YAML file.

        Args:
            save_path (str): Path to the YAML file.
            save_data (Dict): Data to save.
        """
        OmegaConf.save(OmegaConf.create(save_data), save_path)
        logging.info(f"Save {save_path}")  # noqa: G004

    @staticmethod
    def load_mrc(
        load_path: Union[str, Path],
        output_type: type = np.float32,
        get_voxel_size: bool = False,
    ) -> Union[np.ndarray, torch.Tensor, Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]]:
        """Loads data from an MRC file and optionally retrieves voxel size.

        Args:
            load_path (Union[str, Path]):
                Path to the MRC file.
            output_type (type):
                Desired output data type (default=np.float32).
            get_voxel_size (bool):
                If True, returns the voxel size in addition to the data (default=False).

        Returns:
            Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]:
                If `get_voxel_size` is True, returns a tuple containing
                the data loaded from the MRC file and voxel size.

        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        if not Path(load_path).is_file():
            raise FileNotFoundError(f"Error! {load_path} does not exist.")

        with mrcfile.open(load_path, permissive=True) as mrc:
            voxel_size = np.array(
                [mrc.voxel_size.z, mrc.voxel_size.y, mrc.voxel_size.x], dtype=np.float32
            )
            data = mrc.data

        data = (
            torch.from_numpy(data).to(output_type)
            if isinstance(output_type, torch.dtype)
            else data.astype(output_type)
        )

        return (data, voxel_size) if get_voxel_size else data

    @staticmethod
    def save_mrc(
        save_path: Union[str, Path],
        save_data: Union[np.ndarray, torch.Tensor],
        voxel_size: float = 1.0,
        output_type: type = np.float32,
    ) -> None:
        """Saves data to an MRC file.

        Args:
            save_path (Union[str, Path]): Path to save the MRC file.
            save_data (Union[np.ndarray, torch.Tensor]): Data to save.
            voxel_size (float): Voxel size of the data (default=1.0).
            output_type (type): Desired output data type (default=np.float32).
        """
        if isinstance(save_data, torch.Tensor):
            save_data = save_data.detach().cpu().numpy()
        save_data = save_data.astype(output_type)  # type: ignore[union-attr]

        with mrcfile.new(save_path, overwrite=True) as mrc:
            mrc.set_data(save_data)
            mrc.voxel_size = (voxel_size,) * 3
        logging.info(f"Save {save_path}")  # noqa: G004


class ConfigHandler:
    """Provides flexible operations on configuration object.

    Offers features including saving and converting.

    """

    @staticmethod
    def save(cfgs: Union[Dict, DictConfig], file_path: Union[str, Path]) -> None:
        """Saves the configuration as a YAML file, removing the 'tmp' key if present."""
        cfgs_new = copy.deepcopy(cfgs)
        cfgs_new.pop("tmp", None)
        OmegaConf.save(OmegaConf.create(cfgs_new), file_path)

    @staticmethod
    def convert(obj: Union[ListConfig, DictConfig]) -> Union[List, Dict]:
        """Recursively converts ListConfig and DictConfig to standard Python types."""
        if isinstance(obj, (ListConfig, list)):
            return [ConfigHandler.convert(item) for item in obj]
        elif isinstance(obj, (DictConfig, dict)):
            return {key: ConfigHandler.convert(value) for key, value in obj.items()}
        return obj


class CoordsHandler:
    """Handles the conversion of various file formats and transformation of coordinates.

    Provides static methods to convert between STAR and coordinate files,
    extract coordinate data from NDJSON files, calculate centroids from 3D masks, etc.
    """

    @staticmethod
    def star2coords(input_paths: List[Union[str, Path]], output_dir: Union[str, Path]) -> None:
        """Converts STAR files to coordinate files (.coords).

        Args:
            input_paths (List[Union[str, Path]]):
                List of paths to STAR files.
            output_dir (Union[str, Path]):
                Directory to save the resulting coordinate files.
        """
        output_dir = path_resolve(output_dir)
        for input_path in tqdm(input_paths):
            df = starfile.read(input_path)
            df = df.get("particles", df)
            col_name = "rlnMicrographName" if "rlnMicrographName" in df else "rlnTomoName"
            tomo_names = df[col_name].unique()

            for tomo_name in tomo_names:
                df_xyz = df[df[col_name] == tomo_name][
                    ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
                ]

                tomo_name = tomo_name.split("_bin")[0]
                coords_path = output_dir / f"{tomo_name}.coords"
                mod_path = output_dir / f"{tomo_name}.mod"

                df_xyz.to_csv(
                    coords_path,
                    index=False,
                    header=False,
                    float_format="%.2f",
                    sep="\t",
                )
                print(f"Saved {coords_path}")

                if not mod_path.exists():
                    os.system(f"point2model -nu 1 {coords_path} {mod_path}")  # noqa: S605

    @staticmethod
    def coords2star(
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        rlnSphericalAberration: float = 2.7,  # noqa: N803
        rlnVoltage: float = 300.0,  # noqa: N803
        rlnImageSize: int = 0,  # noqa: N803
        rlnImageDimensionality: int = 3,  # noqa: N803
        rlnImagePixelSize: float = 1.0,  # noqa: N803
        name_suffix: str = ".mrc",
        rlnOpticsGroup: int = 1,  # noqa: N803
        split: bool = False,
    ) -> None:
        """Converts coordinate files (.coords or .txt) to STAR files.

        Args:
            input_dir (Union[str, Path]): Directory containing coordinate files.
            output_dir (Union[str, Path]): Directory to save the resulting STAR files.
            rlnSphericalAberration (float): Spherical aberration value (default=2.7).
            rlnVoltage (float): Acceleration voltage in kV (default=300.0).
            rlnImageSize (int): Image size in pixels (default=0).
            rlnImageDimensionality (int): Dimensionality of images (default=3).
            rlnImagePixelSize (float): Pixel size in Ångstrom (default=1.0).
            name_suffix (str): Suffix for micrograph names (default=".mrc").
            rlnOpticsGroup (int): Optics group identifier (default=1).
            split (bool): Whether to split the STAR per coordinate file (default=False).
        """

        def get_df(input_path: Path) -> pd.DataFrame:
            """Creates DataFrame."""
            coords = np.loadtxt(input_path)[:, -3:]
            angle = np.zeros_like(coords)
            df = pd.DataFrame(
                data=np.c_[coords, angle],
                columns=[
                    "rlnCoordinateX",
                    "rlnCoordinateY",
                    "rlnCoordinateZ",
                    "rlnAngleRot",
                    "rlnAngleTilt",
                    "rlnAnglePsi",
                ],
            )
            df["rlnMicrographName"] = input_path.stem + name_suffix
            df["rlnOpticsGroup"] = rlnOpticsGroup
            return df

        optics = pd.DataFrame(
            [
                {
                    "rlnOpticsGroup": rlnOpticsGroup,
                    "rlnOpticsGroupName": f"opticsGroup{rlnOpticsGroup}",
                    "rlnSphericalAberration": rlnSphericalAberration,
                    "rlnVoltage": rlnVoltage,
                    "rlnImageSize": rlnImageSize,
                    "rlnImageDimensionality": rlnImageDimensionality,
                    "rlnImagePixelSize": rlnImagePixelSize,
                }
            ]
        )

        input_dir = path_resolve(input_dir)
        output_dir = path_resolve(output_dir)
        input_paths = sorted(input_dir.glob("*.coords"))

        if split:
            for input_path in input_paths:
                star = {"optics": optics, "particles": get_df(input_path)}
                output_path = output_dir / f"{input_path.stem}.star"
                starfile.write(star, output_path)
                print(f"Saved {output_path}")
        else:
            particles = pd.concat((get_df(p) for p in input_paths), axis=0)
            imgs = pd.DataFrame(
                {
                    "rlnMicrographName": [p.stem + name_suffix for p in input_paths],
                    "rlnOpticsGroup": [rlnOpticsGroup] * len(input_paths),
                }
            )

            particles_star = {"optics": optics, "particles": particles}
            particles_outpath = output_dir / "particles.star"
            starfile.write(particles_star, particles_outpath)
            print(f"Saved {particles_outpath}")

            imgs_star = {"optics": optics, "micrographs": imgs}
            imgs_outpath = output_dir / "micrographs.star"
            starfile.write(imgs_star, imgs_outpath)
            print(f"Saved {imgs_outpath}")

    @staticmethod
    def get_coords_from_ndjson(input_path: Union[str, Path]) -> pd.DataFrame:
        """Extracts coordinate data from the NDJSON file.

        Args:
            input_path (Union[str, Path]): Path to the .ndjson file.

        Returns:
            pd.DataFrame:
                Containing extracted coordinate data with columns "x", "y", and "z".
        """
        with open(input_path, "r") as f:
            data = [
                {"x": loc["x"], "y": loc["y"], "z": loc["z"]}
                for line in f
                if (loc := json.loads(line).get("location", {}))
                and all(k in loc for k in ("x", "y", "z"))
            ]
        return pd.DataFrame(data)

    @staticmethod
    def get_centroids(mask: np.ndarray) -> List:
        """Calculate centroids of labeled regions in a mask.

        Args:
            mask (np.ndarray):
                A 3D binary mask where connected regions will be identified.

        Returns:
            List: List of centroids for each labeled region.
        """
        from skimage.measure import label, regionprops

        props = regionprops(label(mask, connectivity=1))
        return [[prop.centroid[2], prop.centroid[1], prop.centroid[0]] for prop in props] or [
            [0, 0, 0]
        ]


class MaskCoords:
    """Generates masks for 3D coordinates.

    Provides methods to apply binary dilation on a mask,
    generate various 3D structures (such as spheres, cubes, and shells), and
    visualize the results using matplotlib.

    Attributes:
        structure_len (int):
            Length of the structuring element for dilation.
        coords_path (Union[str, Path]):
            Path to the file containing 3D coordinates.
        shape (List):
            Shape of the binary mask.
        types (str):
            Type of structure to be used for dilation ('ball', 'cube', or 'shell').
    """

    def __init__(
        self,
        structure_len: int,
        coords_path: Union[str, Path],
        shape: List,
        types: str = "ball",
    ) -> None:
        """Initializes."""
        super().__init__()
        self.coords_xyz = np.loadtxt(coords_path)
        self.mask = np.zeros(shape)
        self.structure = self.get_structure(structure_len, types=types)

    @staticmethod
    def shell(shell_len: int) -> np.ndarray:
        """Creates a 3D shell structure."""
        from skimage.draw import circle_perimeter

        shell = np.zeros([shell_len] * 3)
        center = shell_len // 2
        for radius in range(center, 0, -1):
            rr, cc = circle_perimeter(center, center, radius)
            shell[radius, rr, cc] = 1
            if center != radius:
                shell[shell_len - 1 - radius, rr, cc] = 1
        return shell

    @staticmethod
    def get_structure(length: int, types: str = "ball") -> np.ndarray:
        """Returns a 3D structure (ball, cube, or shell) based on the specified type."""
        from skimage.morphology import ball, cube

        structure_map = {"ball": ball, "cube": cube, "shell": MaskCoords.shell}

        if types == "ball":
            length = length // 2

        return structure_map[types](length).astype(np.int8)

    @staticmethod
    def coords_on_mask(coords_xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Places coordinates onto the mask."""
        coords_xyz = np.clip(coords_xyz, 0, np.array(mask.shape)[::-1] - 1).astype(np.int16)
        mask[tuple(coords_xyz[:, [2, 1, 0]].T)] = 1
        return mask

    @staticmethod
    def plot(tomo_slice: np.ndarray, masked_slice: np.ndarray) -> None:
        """Visualizes the tomogram slice and masked slice."""
        cmap = ListedColormap(["none", "blue"])
        fig, ax = plt.subplots()
        ax.imshow(tomo_slice, cmap="gray", interpolation="none")
        ax.imshow(masked_slice, cmap=cmap, interpolation="none", alpha=0.5)
        plt.axis("off")
        plt.show()

    def run_cpu(self) -> np.ndarray:
        """Applys binary dilation on the mask using CPU."""
        from scipy.ndimage import binary_dilation as bd

        masked = self.coords_on_mask(self.coords_xyz, self.mask)
        masked = bd(masked, structure=self.structure, iterations=1)
        return masked.astype(np.int8)

    def run_gpu(self, gpu_id: int = 0) -> np.ndarray:
        """Applys binary dilation on the mask using GPU."""
        import cupy as cp
        from cupyx.scipy.ndimage import binary_dilation as cpbd

        masked = self.coords_on_mask(self.coords_xyz, self.mask)
        with cp.cuda.Device(gpu_id):
            masked = cpbd(cp.asarray(masked), structure=cp.asarray(self.structure), iterations=1)
        return cp.asnumpy(masked).astype(np.int8)


class FileManager:
    """Manages file and directory operations.

    Provides methods for ensuring directories exist,
    creating files with version control, etc.

    Example:
        .. code-block:: python

            # Validate file path
            validated_path = FileManager.confirm("example.txt")
            print(validated_path)

            # Create a file with version control
            FileManager.update("new_file.txt")
            # Creates 'new_file.txt' or a versioned file like 'new_file~ver001.txt'

            # Creates the directory if it does not already exist
            FileManager.create("new_directory/")
    """

    @staticmethod
    def confirm(file_path: Union[str, Path]) -> Path:
        """Validates the existence of a file at the given path.

        Args:
            file_path (Union[str, Path]): The file path to validate.

        Returns:
            Path: The resolved Path object of the validated file.

        Raises:
            AssertionError: If the file does not exist or is not a file.
        """
        file_path = Path(file_path)

        if file_path.is_file():
            return file_path.resolve()
        else:
            raise AssertionError(f"Error! File {file_path} does not exist!")

    @staticmethod
    def update(file_path: Union[str, Path], rename_str: str = "~ver") -> None:
        """Creates a file at the given path with version control if needed.

        If a file with the same name exists, this method appends a version number
        (e.g., "~ver001") to the new file's name to avoid overwriting.

        Args:
            file_path (Union[str, Path]):
                The path where the file should be created.
            rename_str (str):
                The versioning suffix to append (default="~ver").
        """
        file_path = Path(file_path)
        if file_path.exists():
            stem, suffix = file_path.stem, file_path.suffix
            similar_files = sorted(file_path.parent.glob(f"{stem}{rename_str}*{suffix}"))
            latest_version = (
                int(similar_files[-1].stem.split(rename_str)[-1]) + 1 if similar_files else 1
            )
            new_file_path = file_path.with_name(f"{stem}{rename_str}{latest_version:03d}{suffix}")
        else:
            new_file_path = file_path

        if "." in file_path.name and file_path.suffix[1:].isalpha():
            new_file_path.touch()
        else:
            new_file_path.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def create(file_path: Union[str, Path]) -> None:
        """Ensure a directory exists at the given path. Create it if necessary.

        If the path includes a file name with an extension, the parent directory
        is created instead.

        Args:
            file_path (Union[str, Path]): The path to the directory or file.
        """
        if file_path:
            file_path = Path(file_path)

            if "." in file_path.name and file_path.suffix[1:].isalpha():
                file_path.parent.mkdir(exist_ok=True, parents=True)
            else:
                file_path.mkdir(exist_ok=True, parents=True)


class StringUtils:
    """Manages string-related operations.

    Provides methods for conversion and comparison.

    Example:
        .. code-block:: python

            # Convert string into a list of integers.
            print(StringUtils.str2int("1,2,3,4"))

            # Convert string into a list of floats.
            print(StringUtils.str2float("1.1,2.2,3.3"))

            # Find differing parts after the longest match
            diff1, diff2 = StringUtils.find_diff_after_match("abcdef", "abcxyz")
            print(diff1, diff2)
    """

    @staticmethod
    def str2int(s: str) -> List[int]:
        """Converts a comma-separated string of integers into a list of integers."""
        return list(map(int, s.split(",")))

    @staticmethod
    def str2float(s: str) -> List[float]:
        """Converts a comma-separated string of numbers into a list of floats."""
        return list(map(float, s.split(",")))

    @staticmethod
    def find_diff_after_match(str1: str, str2: str) -> Tuple[str, str]:
        """Identifies the longest matching substring between two strings.

        Args:
            str1 (str): The first string to compare.
            str2 (str): The second string to compare.

        Returns:
            Tuple: A tuple containing two strings:
                - The part of `str1` after the longest matching substring.
                - The part of `str2` after the longest matching substring.
        """
        from difflib import SequenceMatcher

        # Convert inputs to strings in case they are not
        str1, str2 = str(str1), str(str2)

        # Find the longest match between the two strings
        matcher = SequenceMatcher(None, str1, str2)
        match = matcher.find_longest_match(0, len(str1), 0, len(str2))

        # Extract the differing parts after the longest match
        diff_str1 = str1[match.a + match.size :]
        diff_str2 = str2[match.b + match.size :]

        return diff_str1, diff_str2


class SplitTrainVal:
    """Splits input into training and validation sets."""

    @staticmethod
    def split_data(data: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """Split a 3D array into training and validation sets along its larger dimension.

        Args:
            data (np.ndarray):
                3D numpy array.
            ratio (float):
                Proportion of the larger dimension to assign to the validation set.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                A tuple containing two arrays for validation and training set.
        """
        s0, s1, s2 = data.shape
        if s1 > s2:
            split_index = int(s1 * ratio)
            return data[:, :split_index, :], data[:, split_index:, :]
        else:
            split_index = int(s2 * ratio)
            return data[:, :, :split_index], data[:, :, split_index:]

    @staticmethod
    def split_path(paths: List, indices: List[int]) -> Tuple[List, List]:
        """Splits a list of input paths into training and validation sets.

        Args:
            paths (List): The original list of input paths.
            indices (List[int]): Indices for the validation set.

        Returns:
            Tuple[List, List]:
                A tuple containing two lists for validation and training set.
        """
        val = [paths[i] for i in indices]
        train = [p for p in paths if p not in val]
        return val, train


class Rand:
    """Provides random number generation and sampling methods for reproducibility."""

    @staticmethod
    def uniform(size: Union[int, Tuple[int, ...]], seed: int = 42) -> np.ndarray:
        """Generates random numbers drawn from a uniform distribution.

        Args:
            size (Union[int, Tuple[int, ...]]): The shape of the array to generate.
            seed (int): Seed for reproducibility (default=42).

        Returns:
            np.ndarray: An array of random numbers with the specified shape.
        """
        return np.random.default_rng(seed).random(size)

    @staticmethod
    def sample(
        lst: List[Any],
        size: Optional[int] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Randomly samples elements from a list with a specified seed.

        Args:
            lst (List[Any]): The list to sample from.
            size (Optional[int]): The number of elements to sample (default=None).
            seed (int): Seed for reproducibility (default=42).

        Returns:
            np.ndarray: An array of randomly sampled elements.
        """
        size = size or len(lst)
        rng = np.random.default_rng(seed)
        return rng.choice(lst, size=size, replace=size > len(lst))

    @staticmethod
    def comb(
        lst: List[Any],
        size: Optional[int] = None,
        repeat: int = 3,
        seed: int = 42,
    ) -> List[List]:
        """Generates combinations of elements with repetition.

        Args:
            lst (List[Any]):
                A list of elements to generate combinations from.
            size (Optional[int]):
                Number of combinations to return. If None,
                all possible combinations are returned (default=None).
            repeat (int):
                Number of times elements are repeated in each combination (default=3).
            seed (int):
                Seed for reproducibility (default=42).

        Returns:
            List[List]: Each inner list is a combination of elements.
        """
        rng = np.random.default_rng(seed)
        samples = {tuple(sorted(s)) for s in itertools.product(lst, repeat=repeat)}  # type: ignore[arg-type]
        if size is not None:
            samples = rng.choice(list(samples), size=size, replace=size > len(samples))  # type: ignore[assignment]
        return [list(s) for s in samples]


class BlindMask:
    def __init__(self, roi_len: int, dim: int = 3, use_blind: float = 0.1):
        """Initialize the generator with given parameters."""
        self.roi_size = [roi_len] * dim
        self.probs = BlindMask.get_probs(self.roi_size, use_blind)  # Get probability distribution

    @staticmethod
    def get_distance_squared(roi_size: List[int]) -> torch.Tensor:
        """Generate meshgrid and compute the sum of squared distances from the center."""
        grids = [torch.arange(s) - s // 2 for s in roi_size]
        meshgrids = torch.meshgrid(*grids, indexing="ij")
        distance_sq = sum(g**2 for g in meshgrids)
        return distance_sq

    @staticmethod
    def exponential_decay(roi_size: List[int], lambda_val: float) -> torch.Tensor:
        """Compute a radial probability distribution with exponential decay."""
        distance_sq = BlindMask.get_distance_squared(roi_size)
        distance = torch.sqrt(distance_sq)  # sqrt of squared distance
        return torch.exp(-lambda_val * distance)

    @staticmethod
    def gaussian_decay(roi_size: List[int], sigma_divisor: float) -> torch.Tensor:
        """Compute a radial probability distribution with Gaussian decay."""
        distance_sq = BlindMask.get_distance_squared(roi_size)
        r = min(roi_size) / 2
        sigma = r / sigma_divisor
        return torch.exp(-distance_sq / (2 * sigma**2))

    @staticmethod
    def get_probs(roi_size: List[int], use_blind: float) -> torch.Tensor:
        """Determine which decay function to use based on `use_blind`."""
        if not use_blind or use_blind == 0:
            return torch.ones(roi_size)  # No decay
        elif use_blind < 1:
            return BlindMask.exponential_decay(roi_size, use_blind)
        else:
            return BlindMask.gaussian_decay(roi_size, use_blind)

    def generate(self, seed: int) -> torch.Tensor:
        """Generate a binary blind mask tensor."""
        random_values = torch.rand(
            self.roi_size, generator=torch.Generator().manual_seed(int(seed))
        )
        return (random_values < self.probs).to(torch.bool)


class RandTrans(transforms.Transform, transforms.transform.Randomizable):
    """Performs random transformations (rotation and masking) on 3D data.

    The transformations are applied using random values
    generated for each epoch and sample.

    Attributes:
        max_epochs (int): Number of epochs.
        dataset_len (int): Length of the dataset.
        roi_len (int): Length of each dimension of the ROI (region of interest).
        tilt_angle(float): Tilt angle (in degrees) (default=60).
        seed (int): Seed for random number generation (default=42).
    """

    def __init__(
        self,
        roi_len: int,
        max_epochs: int,
        dataset_len: int,
        tilt_angle: float = 60.0,
        mw_factor: float = 1.5,
        use_blind: int = 3,
        seed: int = 42,
    ) -> None:
        """Initializes."""
        super().__init__()
        mw_angle = 180 - tilt_angle * 2
        self.roi_len = roi_len
        self.mw_func = partial(Mask3D.MissingWedge, shape=[roi_len] * 3)
        self.bm_func = BlindMask(roi_len, dim=3, use_blind=use_blind)
        self.rand_info = self.get_rand_info(max_epochs, dataset_len, mw_angle, mw_factor, seed)

    def __call__(self, d: Dict) -> Dict:
        """Applies a random transformation to the input data dictionary."""
        rot_seed, blind_seed, mw_angle = self.rand_info[d["epoch"]][d["nth"]]

        aff = torch.from_numpy(self.get_rand_affine(rot_seed)[:3, :]).to(torch.float32)
        bm = self.bm_func.generate(blind_seed).unsqueeze(0)
        mw = self.mw_func(mw_angle=mw_angle).unsqueeze(0)

        updates = {"rot_affine": aff, "mask_blind": bm, "mask_mw_fix": mw}
        d.update(updates)
        return d

    @staticmethod
    def get_rand_info(
        max_epochs: int,
        dataset_len: int,
        mw_angle: float = 60.0,
        mw_factor: float = 1.5,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate random arrays for the given number of epochs and dataset length."""
        sample_num = max(max_epochs, 500) * max(dataset_len, 10000)

        rng = np.random.default_rng(seed)

        # Generate independent random seeds for each process
        rot_seeds = rng.choice(
            range(sample_num),
            size=sample_num,
            replace=False,
        ).astype(np.int32)
        blind_seeds = rng.choice(
            range(sample_num),
            size=sample_num,
            replace=False,
        ).astype(np.int32)

        # Generate mw_angles_lst for the angle range
        mw_angles_lst = (
            np.arange(mw_angle, mw_angle * mw_factor + 1, 1.0) if mw_factor > 1 else [mw_angle]
        )
        mw_angles = rng.choice(
            mw_angles_lst,
            size=sample_num,
            replace=True,
        ).astype(np.int32)

        # Stack the generated seeds into one array
        rand_info = np.column_stack(
            [
                rot_seeds,
                blind_seeds,
                mw_angles,
            ]
        ).reshape(max_epochs, -1, 3)

        return rand_info

    @staticmethod
    def get_rand_affine(seed: int = 42) -> np.ndarray:
        """Generate a uniformly random 3D rotation affine matrix."""
        rotation = R.random(random_state=seed)
        affine_matrix_3d = np.eye(4)
        affine_matrix_3d[:3, :3] = rotation.as_matrix()
        return affine_matrix_3d

    @staticmethod
    def get_train_batch(
        d: Dict,
        mask_sphere: torch.Tensor,
        mask_mw: torch.Tensor,
        tomo_keys: List[str] = ["all"],  # noqa: B006
    ) -> Dict:
        """Process a batch of data, applies rotation and masking.

        Padding by one voxel (making the size odd) aligns the
        rotation center correctly; after rotation,
        the extra voxel is removed to restore the original shape.

        Args:
            d (Dict):
                A dictionary containing the batch of data.
            mask_sphere (torch.Tensor):
                A binary mask representing the spherical region.
            mask_mw (torch.Tensor):
                A binary mask representing the missing wedge.
            tomo_keys (List[str]):
                Keys to access and update the data items
                in the input dictionary (default=["all"]).

        Returns:
            Dict: A dictionary containing the updated batch.
        """

        def pad_func(x):
            return F.pad(x, (0, 1, 0, 1, 0, 1), mode="constant", value=0)

        def ipad_func(x):
            return x[..., :-1, :-1, :-1]

        ori_shape = mask_sphere.shape
        target_shape = ori_shape[2:]
        pad_shape = list(ori_shape[:2]) + [s + 1 for s in target_shape]

        crop_func = partial(DataHandler.crop_center, target_shape=target_shape)
        rot_img_func = partial(
            F.grid_sample,
            mode="bilinear",
            padding_mode="reflection",
            grid=F.affine_grid(d["rot_affine"], pad_shape),
        )
        rot_mask_func = partial(
            F.grid_sample,
            mode="nearest",
            padding_mode="zeros",
            grid=F.affine_grid(d["rot_affine"], pad_shape),
        )

        d["mask_mw_rot"] = (
            ipad_func(rot_mask_func(pad_func(mask_mw.to(torch.float32)))) * mask_sphere
        ).to(torch.bool)
        d["mask_mw_fix"] = (d["mask_mw_fix"] * mask_sphere).to(torch.bool)

        for tomo_key in tomo_keys:  # type: ignore[union-attr]
            if tomo_key in d:
                x = d[tomo_key].to(torch.float32)
                x = ipad_func(rot_img_func(pad_func(x)))
                d[f"{tomo_key}_rot"] = crop_func(x)

        return d

    @staticmethod
    def get_rot_affine_fixed(angle: float, axis: Optional[int] = None) -> np.ndarray:
        """Generates a 3D rotation affine matrix based on the specified angle and axis.

        Args:
            angle (float): The rotation angle in degrees.
            axis (Optional[int]): The axis of rotation, specified as an integer
                from 0 to 11, each representing a unique combination of rotations
                around the y, x, and z axes. If None, a default rotation around
                the y-axis is applied  (default=None).

        Returns:
            np.ndarray: A 4x4 affine matrix representing the specified rotation.

        Note:
            Matrix multiplication `@` is order-sensitive:
            `rot_2 @ rot_1` means applying `rot_1` first, followed by `rot_2`.
            The axis of rotation is relative to itself.
        """
        axis_maps = {
            0: ("y", 90, "x"),
            1: ("y", 270, "x"),
            2: ("y", 90, "z"),
            3: ("y", 270, "z"),
            4: ("x", 90, "z"),
            5: ("x", 270, "z"),
            6: ("x", 90, "y"),
            7: ("x", 270, "y"),
            8: ("z", 90, "y"),
            9: ("z", 270, "y"),
            10: ("z", 90, "x"),
            11: ("z", 270, "x"),
        }  # y (0-3, base: top2); x (4-7); z (8-11)

        affine_matrix_3d = np.eye(4)
        if axis in axis_maps:
            axis_1, angle_1, axis_2 = axis_maps[axis]
            rot_1 = R.from_euler(axis_1, np.deg2rad(angle_1)).as_matrix()
            rot_2 = R.from_euler(axis_2, np.deg2rad(angle)).as_matrix()
            affine_matrix_3d[:3, :3] = rot_2 @ rot_1
        else:
            # The MW will appear in the x-z plane.
            affine_matrix_3d[:3, :3] = R.from_euler("y", np.deg2rad(angle)).as_matrix()

        return affine_matrix_3d


class LoadTomo(transforms.Transform):
    """A custom PyTorch transform to load MRC files.

    This transform reads MRC files specified by the keys in the input dictionary,
    optionally adds a channel dimension, and updates the dictionary
    with the loaded volumes.

    Attributes:
        keys (List):
            Keys to access and update the data items in the input dictionary.
        add_channel (bool):
            Whether to add a channel dimension to the loaded data (default=True).
        output_type (List):
            Desired output data types for each key (default=None).

    Raises:
        ValueError: If the input file does not exist.
    """

    def __init__(
        self,
        keys: List,
        add_channel: bool = True,
        output_type: Optional[List] = None,
    ) -> None:
        """Initializes."""
        self.keys = keys
        self.add_channel = add_channel
        self.output_type = output_type or [torch.float32] * len(keys)

    def __call__(self, d: Dict, split_suffix: str = "_bin") -> Dict:
        """Loads MRC files for each key."""
        for key, output_type in zip(self.keys, self.output_type, strict=False):
            mrc_path = Path(d.get(key, f"Error! {key} does not exist."))
            volume = FileHandler.load_mrc(mrc_path, output_type)
            d[key] = volume.unsqueeze(0) if self.add_channel else volume  # type: ignore[union-attr]
            d.setdefault("name", mrc_path.stem.split(split_suffix)[0])

        return d


class ApplyFourier(transforms.Transform):
    """Applies Fourier transform with masking to specified keys in input data.

    Provides methods to perform 3D FFT and IFFT operations, apply Fourier masking,
    compute magnitude and phase of complex tensors, etc.

    Attributes:
        keys (List):
            Keys to access and update the data items in the input dictionary.
        tomo (str):
            Key for the volume to be transformed.
        mask (str):
            Key for the mask to be applied during the Fourier transform.
    """

    def __init__(self, keys: List, tomo: str, mask: str) -> None:
        """Initializes."""
        self.keys = keys
        self.tomo = tomo
        self.mask = mask

    @staticmethod
    def fft_3d(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
        """Applies 3D Fast Fourier Transform (FFT) to input data."""
        dim = tuple(range(2, x.ndim))
        return torch.fft.fftshift(torch.fft.fftn(x, dim=dim, norm=norm), dim=dim)

    @staticmethod
    def ifft_3d(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
        """Applies Inverse Fast Fourier Transform (IFFT) to input data."""
        dim = tuple(range(2, x.ndim))
        return torch.fft.ifftn(torch.fft.ifftshift(x, dim=dim), dim=dim, norm=norm)

    @staticmethod
    def apply_fourier_mask_to_tomo(
        tomo: torch.Tensor, mask: torch.Tensor, output: str = "real"
    ) -> torch.Tensor:
        """Applies Fourier transform to volume with masking."""
        tomo = ApplyFourier.fft_3d(tomo) * mask
        tomo = ApplyFourier.ifft_3d(tomo)
        return tomo.real if output == "real" else tomo

    def __call__(self, d: Dict) -> Dict:
        """Perform Fourier transform for each key."""
        for key in self.keys:
            d[key] = self.apply_fourier_mask_to_tomo(d[self.tomo], d[self.mask])
        return d

    @staticmethod
    def get_mag_pha(
        x: torch.Tensor, to_log: bool = True, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the magnitude and phase of a complex tensor.

        Args:
            x (torch.Tensor):
                A complex tensor.
            to_log (bool):
                Whether to apply a logarithmic transformation
                to the magnitude (default=True).
            epsilon (float):
                Threshold to zero out small real and
                imaginary parts for stability (default=1e-6).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Magnitude and phase of the input tensor.
        """
        real_part = x.real
        imag_part = x.imag

        real_part = torch.where(
            torch.abs(real_part) < epsilon,
            torch.tensor(0.0, dtype=real_part.dtype),
            real_part,
        )
        imag_part = torch.where(
            torch.abs(imag_part) < epsilon,
            torch.tensor(0.0, dtype=imag_part.dtype),
            imag_part,
        )

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        if to_log:
            magnitude = torch.log1p(magnitude)  # log(1 + magnitude)
        phase = torch.atan2(imag_part, real_part)
        return magnitude, phase

    @staticmethod
    def get_restored_complex(
        magnitude: torch.Tensor, phase: torch.Tensor, from_log: bool = True
    ) -> torch.Tensor:
        """Reconstructs a complex tensor from its magnitude and phase.

        Args:
            magnitude (torch.Tensor):
                The magnitude of the complex numbers.
            phase (torch.Tensor):
                The phase of the complex numbers.
            from_log (bool):
                Indicates if the magnitude is in logarithmic form (default=True).

        Returns:
            torch.Tensor: A tensor of complex numbers.
        """
        if from_log:
            magnitude = torch.expm1(
                magnitude
            )  # Convert from log scale if necessary: exp(log_magnitude) - 1
        restored_real = magnitude * torch.cos(phase)
        restored_imag = magnitude * torch.sin(phase)
        return torch.complex(restored_real, restored_imag)


class Mask3D:
    """Provides methods for creating and manipulating 3D binary masks."""

    @staticmethod
    def post(mask: torch.Tensor) -> torch.Tensor:
        """Post-processes mask by setting certain boundaries to zero based on its shape.

        For even-shaped masks, clears the boundary layers to avoid edge effects,
        ensuring proper frequency domain symmetry.

        Args:
            mask (torch.Tensor): A 3D tensor representing the mask.

        Returns:
            torch.Tensor: The modified mask after post-processing.
        """
        if mask.shape[0] % 2 == 0:
            mask[[0], :, :] = 0
            mask[:, [0], :] = 0
            mask[:, :, [0]] = 0
        return mask

    @staticmethod
    def insert(input_mask: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
        """Inserts the `input_mask` into the center of a new tensor.

        Args:
            input_mask (torch.Tensor):
                A 3D tensor to be inserted.
            output_shape (Sequence[int]):
                The desired shape of the output tensor.

        Returns:
            torch.Tensor: A tensor with the input_mask centered within it.
        """
        input_shape = input_mask.shape
        insert_position = [(o - i) // 2 for o, i in zip(output_shape, input_shape, strict=False)]
        slices = tuple(slice(p, p + s) for p, s in zip(insert_position, input_shape, strict=False))

        output_mask = torch.zeros(output_shape, dtype=torch.uint8, device=input_mask.device)
        output_mask[slices] = input_mask
        return output_mask

    @staticmethod
    def MissingWedge(  # noqa: N802
        shape: Sequence[int],
        mw_angle: float = 60,
        device: str = "cpu",
        post: bool = True,
    ) -> torch.Tensor:
        """Create a 3D binary mask representing the missing wedge in its frequency domain.

        Args:
            shape (Sequence[int]):
                The shape of the mask.
            mw_angle (float):
                The angle (in degrees) that defines the missing wedge (default=60).
            device (str):
                The device to run the computation on (default="cpu").
            post (bool):
                A flag indicating whether to apply post-processing on the generated mask
                (default=True).

        Returns:
            torch.Tensor: A 3D binary mask representing the missing wedge.

        Note:
            Adapted from:
            https://github.com/MLI-lab/DeepDeWedge/blob/master/ddw/utils/missing_wedge.py
        """
        grid_size = [int(s) for s in shape]
        tensors = [torch.fft.fftshift(torch.fft.fftfreq(s, device=device)) for s in grid_size]
        grid = torch.cartesian_prod(*tensors)

        alpha = torch.deg2rad(torch.tensor(float(mw_angle))) / 2
        normal_left = torch.tensor([torch.sin(alpha), 0, torch.cos(alpha)], device=device)
        normal_right = torch.tensor([torch.sin(alpha), 0, -torch.cos(alpha)], device=device)

        upper_wedge = torch.logical_or(grid @ normal_left >= 0, grid @ normal_right >= 0).reshape(
            grid_size
        )
        lower_wedge = torch.logical_or(grid @ normal_left <= 0, grid @ normal_right <= 0).reshape(
            grid_size
        )
        mw_mask = torch.logical_and(upper_wedge, lower_wedge).to(torch.bool)

        if post:
            return Mask3D.post(mw_mask)
        else:
            return mw_mask

    @staticmethod
    def Sphere(  # noqa: N802
        shape: Sequence[int],
        radius: Optional[int] = None,
        device: str = "cpu",
        post: bool = True,
    ) -> torch.Tensor:
        """Creates a 3D binary mask representing a sphere.

        The sphere is defined by its radius. If the radius is not provided,
        it defaults to half of the smallest dimension of the given shape.

        Args:
            shape (Sequence[int]): Shape of the mask.
            radius (Optional[int]): Radius of the sphere (default=None).
            device (str): Device to run the computation on (default="cpu").
            post (bool):
                A flag indicating whether to apply post-processing on the generated mask
                (default=True).

        Returns:
            torch.Tensor: A 3D binary mask representing a sphere.
        """
        radius = radius or min(shape) // 2

        grids = [torch.arange(s, device=device) - s // 2 for s in shape]
        meshgrids = torch.meshgrid(*grids, indexing="ij")

        distance = torch.sqrt(sum(g**2 for g in meshgrids))
        mask = (distance <= radius).to(torch.bool)

        if post:
            return Mask3D.post(mask)
        else:
            return mask


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss for Image Reconstruction and Synthesis.

    A frequency domain loss function for optimizing generative models.

    Attributes:
        loss_weight (float):
            Weight for focal frequency loss (default=1.0).
        alpha (float):
            Scaling factor for the spectrum weight matrix (default=1.0).
        log_matrix (bool):
            Adjust spectrum weight matrix by logarithm (default=False).
        batch_matrix (bool):
            Calculate spectrum weight matrix
            using batch-based statistics (default=False).

    Note:
        Adapted from:
        https://github.com/EndlessSora/focal-frequency-loss/blob/master/focal_frequency_loss/focal_frequency_loss.py
        Expand the original function from 2D to 3D.

    Reference:
        Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
        <https://arxiv.org/pdf/2012.12821.pdf>
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        log_matrix: bool = False,
        batch_matrix: bool = False,
    ) -> None:
        """Initializes."""
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    @staticmethod
    def get_dist(diff: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Calculate the distance using a power of the absolute difference."""
        return torch.abs(diff).pow(alpha)

    @staticmethod
    def tensor2freq(x: torch.Tensor) -> torch.Tensor:
        """Converts a tensor from the spatial domain to the frequency domain."""
        dim = tuple(range(2, x.ndim))
        return torch.fft.fftn(x, dim=dim, norm="ortho")

    def get_matrix(self, pred_freq: torch.Tensor, target_freq: torch.Tensor) -> torch.Tensor:
        """Calculates the spectrum weight matrix based on the Euclidean distance."""
        matrix = self.get_dist(pred_freq - target_freq, self.alpha)

        # Apply logarithmic scaling if required
        if self.log_matrix:
            matrix = torch.log1p(matrix)

        # Normalize the matrix
        if self.batch_matrix:
            matrix /= matrix.max()
        else:
            dim = tuple(range(2, target_freq.ndim))
            max_vals = matrix.amax(dim=dim, keepdim=True)
            matrix /= max_vals

        # Handle NaN values and clamp the matrix
        matrix = torch.nan_to_num(matrix, nan=0.0)
        matrix = torch.clamp(matrix, min=0.0, max=1.0)
        return matrix.clone().detach()

    def loss_formulation(
        self,
        pred_freq: torch.Tensor,
        target_freq: torch.Tensor,
        matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates the frequency domain loss with dynamic spectrum weighting.

        Args:
            pred_freq (torch.Tensor):
                Predicted tensor in the frequency domain.
            target_freq (torch.Tensor):
                Target tensor in the frequency domain.
            matrix (Optional[torch.Tensor]):
                Element-wise spectrum weight matrix (default=None).

        Returns:
            torch.Tensor: Mean of the weighted frequency domain loss.
        """
        matrix = matrix.detach() if matrix is not None else self.get_matrix(pred_freq, target_freq)
        dist = self.get_dist(pred_freq - target_freq, alpha=2)
        loss = matrix * dist  # Dynamic spectrum weighting (Hadamard product)
        return torch.mean(loss)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor):
                Predicted tensor in the spatial domain.
            target (torch.Tensor):
                Target tensor in the spatial domain.
            matrix (Optional[torch.Tensor]):
                Element-wise spectrum weight matrix (default=None).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight


class MetricsCalculator:
    """Calculate various metrics between ground truth and predicted tensors.

    Provides methods to compute metrics such as Signal-to-Noise Ratio (SNR),
    Correlation Coefficient (CC), Log Frequency Distance (LFD),
    Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM),
    Multi-Scale Structural Similarity Index (MSSSIM), etc.

    Attributes:
        true (Union[np.ndarray, torch.Tensor]): Ground truth data.
        pred (Union[np.ndarray, torch.Tensor]): Predicted data.
        device (str): The device to perform computations on (default="cpu").
    """

    def __init__(
        self,
        true: Union[np.ndarray, torch.Tensor],
        pred: Union[np.ndarray, torch.Tensor],
        device: str = "cpu",
    ) -> None:
        """Initializes."""
        scale_func = partial(DataHandler.scale_tensor, scale=True)
        true = (
            torch.tensor(true, dtype=torch.float32, device=device)
            if not isinstance(true, torch.Tensor)
            else true.to(device)
        )
        pred = (
            torch.tensor(pred, dtype=torch.float32, device=device)
            if not isinstance(pred, torch.Tensor)
            else pred.to(device)
        )

        # BCDHW
        dim = true.dim()
        if dim in [3, 4]:
            true = true.unsqueeze(0).unsqueeze(0) if dim == 3 else true.unsqueeze(1)
            pred = pred.unsqueeze(0).unsqueeze(0) if dim == 3 else pred.unsqueeze(1)

        # 0-1
        self.true = scale_func(true)
        self.pred = scale_func(pred)

    @staticmethod
    def CC(  # noqa: N802
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        reduction: str = "mean_batch",
        epsilon: float = 1e-6,
    ) -> Union[float, torch.Tensor]:
        """Calculate the zero-mean correlation coefficient for batches of 2D, 3D images.

        The correlation coefficient is defined as:

        .. math::

            CC(y_{true}, y_{pred}) =
            \\frac{\\sum_i (y_{true,i} - \\bar{y}_{true}) (y_{pred,i} - \\bar{y}_{pred})}
                 {\\sqrt{\\sum_i (y_{true,i} - \\bar{y}_{true})^2 } \\,
                  \\sqrt{\\sum_i (y_{pred,i} - \\bar{y}_{pred})^2 }}

        where :math:`\\bar{y}_{true}` and :math:`\\bar{y}_{pred}` are the spatial means
        of the ground truth and prediction tensors, respectively.

        Args:
            y_true (torch.Tensor):
                Ground truth tensor, shape (B, C, ...).
            y_pred (torch.Tensor):
                Predicted tensor, shape (B, C, ...).
            reduction (str):
                Reduction method:
                  - "mean_batch": return mean scalar across batch (default).
                  - otherwise: return per (B, C) tensor.
            epsilon (float):
                Small value to prevent division by zero (default=1e-6).

        Returns:
            Union[float, torch.Tensor]: The calculated correlation coefficient.
                A higher value is preferable, with 1 being ideal.
        """
        dim = tuple(range(2, y_true.ndim))

        # Subtract mean along spatial dims
        zmean1 = y_true - y_true.mean(dim=dim, keepdim=True)
        zmean2 = y_pred - y_pred.mean(dim=dim, keepdim=True)

        # Compute norms
        norm1 = torch.sqrt(torch.sum(zmean1**2, dim=dim)).clamp_min(epsilon)
        norm2 = torch.sqrt(torch.sum(zmean2**2, dim=dim)).clamp_min(epsilon)

        # Correlation per sample-channel
        correlation = torch.sum(zmean1 * zmean2, dim=dim) / (norm1 * norm2)

        # Detach for metric usage
        value = correlation.detach().cpu()

        return value.mean().item() if reduction == "mean_batch" else value

    @staticmethod
    def LFD(  # noqa: N802
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        reduction: str = "mean_batch",
    ) -> Union[float, torch.Tensor]:
        """Calculate the Log Frequency Distance (LFD) for batches of 2D, 3D images.

        The Log Frequency Distance is defined as:

        .. math::

            LFD(y_{true}, y_{pred}) =
            \\log\\left( \\frac{1}{N} \\sum_{i \\in \\Omega}
            \\Big[ (\\Re(F_{true,i}) - \\Re(F_{pred,i}))^2 +
                  (\\Im(F_{true,i}) - \\Im(F_{pred,i}))^2 \\Big] + 1 \\right)

        where:

        - :math:`F_{true} = \\mathcal{F}(y_{true})` and
          :math:`F_{pred} = \\mathcal{F}(y_{pred})` are the multi-dimensional
          discrete Fourier transforms of the ground truth and predicted tensors,
          respectively.
        - :math:`\\Re(\\cdot)` and :math:`\\Im(\\cdot)` denote the real and
          imaginary parts.
        - :math:`\\Omega` is the set of all spatial frequency indices.
        - :math:`N` is the number of elements in :math:`\\Omega`.

        Args:
            y_true (torch.Tensor):
                Ground truth tensor, shape (B, C, ...).
            y_pred (torch.Tensor):
                Predicted tensor, shape (B, C, ...).
            reduction (str):
                Reduction method:
                  - "mean_batch": return mean scalar across batch (default).
                  - otherwise: return per (B, C) tensor.

        Returns:
            Union[float, torch.Tensor]:
                The calculated LFD value.
                A lower value is preferable, with 0 being ideal.

        Note:
            Adapted from:
            https://github.com/EndlessSora/focal-frequency-loss/blob/master/metrics/metric_utils.py
            Expand the original function from 2D to 3D.
        """
        dim = tuple(range(2, y_true.ndim))

        # Compute multi-dimensional Fourier transforms
        freq1 = torch.fft.fftn(y_true, dim=dim)  # , norm="ortho"
        freq2 = torch.fft.fftn(y_pred, dim=dim)  # , norm="ortho"

        # Frequency differences (real and imaginary parts)
        diff_real = (freq1.real - freq2.real) ** 2
        diff_imag = (freq1.imag - freq2.imag) ** 2

        # Log frequency distance:
        #   LFD = log( mean(diff_real + diff_imag) + 1 )
        value = torch.log(torch.mean(diff_real + diff_imag, dim=dim) + 1.0).detach().cpu()

        return value.mean().item() if reduction == "mean_batch" else value

    @staticmethod
    def SNR(  # noqa: N802
        noisys: torch.Tensor,
        masks: torch.Tensor,
        iterations: Optional[int] = None,
        reduction: str = "mean_batch",
    ) -> Union[float, torch.Tensor]:
        """Calculate the Signal-to-Noise Ratio (SNR) or Contrast-to-Noise Ratio (CNR)
        for batches of 3D images.

        Args:
            noisys (torch.Tensor):
                Input noisy tensor batch, shape (B, C, D, H, W).
            masks (torch.Tensor):
                Boolean mask tensor batch indicating signal regions, same shape as `noisys`.
            iterations (Optional[int]):
                Number of dilation iterations to define local background region.
                - If None, compute SNR.
                - If provided, compute CNR using local neighborhood.
            reduction (str):
                Reduction method for batch output:
                - "mean_batch": return mean scalar across batch (default).
                - otherwise: return per-sample tensor of shape (B,).

        Returns:
            Union[float, torch.Tensor]:The computed SNR or CNR value(s).
                A higher value is preferable.
        """

        def dilation_3d(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
            """Perform binary dilation on a 3D boolean mask."""
            # Ensure shape is (1, 1, D, H, W)
            while mask.dim() < 5:
                mask = mask.unsqueeze(0)

            mask = mask.float()
            for _ in range(iterations):
                mask = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
            return (mask > 0).squeeze(0)

        def compute_snr(noisy: torch.Tensor, mask: torch.Tensor) -> float:
            """Compute variance-based SNR from a noisy volume and a foreground mask.

            SNR is computed as:
                $\\mathrm{SNR} = \frac{\\sigma^2_{\text{signal}}}{\\sigma^2_{\text{noise}}}$.
            """
            mask = mask.bool()

            signal_region = noisy[mask]
            noise_region = noisy[~mask]

            if signal_region.numel() == 0 or noise_region.numel() == 0:
                return 0.0

            var_signal_noisy = torch.var(signal_region, unbiased=False)
            var_noise = torch.var(noise_region, unbiased=False)

            snr_linear = (var_signal_noisy - var_noise) / var_noise
            return snr_linear.item()

        def compute_cnr(
            noisy: torch.Tensor,
            mask: torch.Tensor,
            iterations: int = 1,
        ) -> float:
            """Compute CNR based on local neighborhood dilation.

            CNR is computed as:
                $\\mathrm{CNR} = \frac{\\mu_s - \\mu_n}{\\sigma_n}$.
            """
            mask = mask.bool()

            # Get signal region
            signal_region = noisy[mask]

            # Dilate mask to get local background shell
            dilated_mask = dilation_3d(mask, iterations=iterations)
            shell_mask = dilated_mask & (~mask)
            noise_region = noisy[shell_mask]

            if signal_region.numel() == 0 or noise_region.numel() == 0:
                return 0.0

            mu_s = signal_region.mean()
            mu_n = noise_region.mean()
            sigma_n = torch.std(noise_region, unbiased=False)

            cnr = (mu_s - mu_n) / sigma_n
            return cnr.item()

        def fn(
            noisy: torch.Tensor,
            mask: torch.Tensor,
            iterations: Optional[int] = None,
            to_db: bool = False,
        ) -> float:
            """Compute SNR or CNR for one sample."""
            if iterations is None:
                var_linear = compute_snr(noisy, mask)
                # Convert to dB
                if to_db:
                    return 10 * math.log10(var_linear)

            else:
                var_linear = compute_cnr(noisy, mask, iterations)

            return var_linear

        # Ensure shape is (1, 1, D, H, W)
        while noisys.dim() < 5:
            noisys = noisys.unsqueeze(0)
        while masks.dim() < 5:
            masks = masks.unsqueeze(0)

        value = [fn(noisy, mask, iterations) for noisy, mask in zip(noisys, masks, strict=False)]
        value = torch.tensor(value)  # type: ignore[assignment]
        return value.mean().item() if reduction == "mean_batch" else value  # type: ignore[attr-defined,return-value]

        # noisy = DataHandler.scale_tensor(-noisy, scale=True)

    @staticmethod
    def PSNR(  # noqa: N802
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        reduction: str = "mean_batch",
    ) -> Union[float, torch.Tensor]:
        """Compute Peak Signal-to-Noise Ratio (PSNR) for batched 2D or 3D images.

        Args:
            y_true (torch.Tensor):
                Ground-truth tensor of shape (B, C, ...).
            y_pred (torch.Tensor):
                Predicted tensor of shape (B, C, ...).
            reduction (str):
                Reduction method:
                - "mean_batch": return scalar mean over batch.
                - otherwise: return per-batch values of shape (B,).

        Returns:
            Union[float, torch.Tensor]: The calculated PSNR value.
                A higher value is preferable.
        """
        from torchmetrics.functional.image import peak_signal_noise_ratio

        def fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute PSNR for one sample."""
            return peak_signal_noise_ratio(pred, target, data_range=1.0)

        value = torch.vmap(fn, in_dims=(0, 0))(y_pred, y_true)
        return value.mean().item() if reduction == "mean_batch" else value

    @staticmethod
    def SSIM(  # noqa: N802
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        reduction: str = "mean_batch",
    ) -> Union[float, torch.Tensor]:
        """Compute Structural Similarity Index Measure (SSIM) for batched 2D or 3D images.

        Args:
            y_true (torch.Tensor):
                Ground-truth tensor of shape (B, C, ...).
            y_pred (torch.Tensor):
                Predicted tensor of shape (B, C, ...).
            reduction (str):
                Reduction method:
                - "mean_batch": return scalar mean over batch.
                - otherwise: return per-batch values of shape (B,).

        Returns:
            Union[float, torch.Tensor]: The calculated SSIM value.
                A higher value is preferable.
        """
        from torchmetrics.functional.image import structural_similarity_index_measure

        def fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute SSIM for one sample."""
            return structural_similarity_index_measure(pred, target, data_range=1.0)

        value = torch.tensor([fn(p, t) for p, t in zip(y_pred, y_true, strict=False)])

        return value.mean().item() if reduction == "mean_batch" else value

    @staticmethod
    def PSNR_monai(  # noqa: N802
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        max_val: float = 1.0,  # be careful
        reduction: str = "mean_batch",
    ) -> Union[float, torch.Tensor]:
        """Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

        Args:
            y_true (torch.Tensor): Ground truth tensor with shape BCDHW.
            y_pred (torch.Tensor): Predicted tensor with shape BCDHW.
            max_val: The maximum possible pixel value of the images (default=1.0).
            reduction (str): Reduction method (default="mean_batch").

        Returns:
            Union[float, torch.Tensor]:
                The PSNR value as a float. If reduction is "none", returns a tensor;
                otherwise, returns a scalar. A higher value is preferable.
        """
        cal_func = PSNRMetric(max_val, reduction=reduction)  # type: ignore[arg-type]
        cal_func(y_true, y_pred)
        value = cal_func.aggregate().detach().cpu()  # type: ignore[union-attr]

        return value.mean().item() if reduction == "mean_batch" else value

    @staticmethod
    def SSIM_monai(  # noqa: N802
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        spatial_dims: int = 3,
        data_range: float = 1.0,
        win_size: int = 11,
        reduction: str = "mean_batch",
    ) -> Union[float, torch.Tensor]:
        """Calculates the Structural Similarity Index (SSIM) between two tensors.

        Args:
            y_true (torch.Tensor):
                Ground truth tensor with shape BCDHW.
            y_pred (torch.Tensor):
                Predicted tensor with shape BCDHW.
            spatial_dims (int):
                Number of spatial dimensions (default=3).
            data_range (float):
                Range of the data (default=1.0).
            win_size (int):
                Size of the window for SSIM calculation (default=11).
            reduction (str):
                Reduction method (default="mean_batch").

        Returns:
            Union[float, torch.Tensor]:
                SSIM value as a scalar if reduction is not "none", otherwise as a tensor.
                A higher value is preferable, with 1 being ideal.
        """
        cal_func = SSIMMetric(
            spatial_dims=spatial_dims,  # type: ignore[arg-type]
            data_range=data_range,  # type: ignore[arg-type]
            win_size=win_size,  # type: ignore[arg-type]
            reduction=reduction,  # type: ignore[arg-type]
        )
        cal_func(y_true, y_pred)
        value = cal_func.aggregate().detach().cpu()  # type: ignore[union-attr]

        return value.mean().item() if reduction == "mean_batch" else value

    @staticmethod
    def MSSSIM(  # noqa: N802
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        spatial_dims: int = 3,
        data_range: float = 1.0,
        kernel_size: int = 11,
        reduction: str = "mean_batch",
    ) -> Union[float, torch.Tensor]:
        """Calculates the Multi-Scale Structural Similarity Index (MS-SSIM).

        Args:
            y_true (torch.Tensor): Ground truth tensor with shape BCDHW.
            y_pred (torch.Tensor): Predicted tensor with shape BCDHW.
            spatial_dims (int): Number of spatial dimensions (default=3).
            data_range (float): Data range of the input images (default=1.0).
            kernel_size (int): Size of the Gaussian kernel (default=11).
            reduction (str): Reduction method (default="mean_batch").

        Returns:
            Union[float, torch.Tensor]:
                MS-SSIM value as a scalar if reduction is not "none",
                otherwise as a tensor. A higher value is preferable, with 1 being ideal.
        """
        cal_func = MultiScaleSSIMMetric(
            spatial_dims=spatial_dims,  # type: ignore[arg-type]
            data_range=data_range,  # type: ignore[arg-type]
            kernel_size=kernel_size,  # type: ignore[arg-type]
            reduction=reduction,  # type: ignore[arg-type]
        )
        cal_func(y_true, y_pred)
        value = cal_func.aggregate().detach().cpu()  # type: ignore[union-attr]

        return value.mean().item() if reduction == "mean_batch" else value

    def get_metric(self, metrics: List) -> Dict:
        """Retrieves specified metrics for the tensors.

        Args:
            metrics (List):
                List of strings representing the names of the
                metric methods to be calculated.

        Returns:
            Dict:
                A dictionary where keys are metric names and
                values are the computed metric values.
        """
        return {metric: getattr(self, metric)(self.true, self.pred) for metric in metrics}


class TrainUtils:
    """Utility class for training-related system configurations and GPU management.

    Provides methods for displaying system configurations, managing GPU settings,
    calculating training steps, and monitoring GPU memory usage.

    Example:
        .. code-block:: python

            # Show system configuration
            TrainUtils.show_system_config()

            # Set the active GPU to GPU #0
            TrainUtils.set_active_gpu(0)

            # Calculate total training steps
            steps = TrainUtils.calculate_total_steps(
                num_data=10000,
                batch_size_per_gpu=16,
                max_epochs=10,
                num_gpus=2
            )
            print("Total Training Steps:", steps)

            # Monitor GPU memory and wait until enough memory is available
            TrainUtils.monitor_gpu_memory(threshold=24000, interval=5, gpu_index=0)
    """

    @staticmethod
    def show_system_config() -> None:
        """Displays the system and library configuration.

        Uses `scooby` and `monai` to print a detailed system configuration report.
        """
        import scooby

        print(scooby.Report())
        print_config()

    @staticmethod
    def set_active_gpu(gpu_id: int) -> None:
        """Sets the active GPU by ID and logs the GPU information."""
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")

        for idx in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(idx)
            print(f"     GPU #{idx}: {gpu_name}")

        if gpu_count > 0 and 0 <= gpu_id < gpu_count:
            torch.cuda.set_device(gpu_id)
        if torch.cuda.is_available():
            print(f"Currently using: {torch.cuda.get_device_name(gpu_id)}")
            print("GPU is available.")
        else:
            print("Invalid GPU ID or no GPU available.")

    @staticmethod
    def calculate_total_steps(
        num_data: int, batch_size_per_gpu: int, max_epochs: int, num_gpus: int
    ) -> int:
        """Calculates the total number of training steps.

        Args:
            num_data (int): The total number of data points in the dataset.
            batch_size_per_gpu (int): Batch size per GPU.
            max_epochs (int): The total number of training epochs.
            num_gpus (int): The number of GPUs used for training.

        Returns:
            int: The total number of training steps.
        """
        total_batch_size = batch_size_per_gpu * num_gpus
        steps_per_epoch = math.ceil(num_data / total_batch_size)
        return steps_per_epoch * max_epochs

    @staticmethod
    def monitor_gpu_memory(
        threshold: int = 24000,
        interval: float = 10,
        gpu_index: int = 0,
    ) -> None:
        """Monitor GPU memory and waits until the available memory exceeds the threshold.

        Args:
            threshold (int):
                Minimum required available GPU memory in MiB (defaults=24000).
            interval (float):
                Time in seconds between memory checks (default=10).
            gpu_index (int):
                Index of the GPU to monitor (default=0).
        """
        import pynvml

        while True:
            if torch.cuda.is_available():
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_mem = mem_info.free / 1024**2
                total_mem = mem_info.total / 1024**2

                info = f"Available GPU Mem: {free_mem:.2f} MiB /{total_mem:.2f} MiB."
                if free_mem > threshold:
                    info += " Proceeding..."
                    logging.info(info)
                    pynvml.nvmlShutdown()
                    break
                else:
                    info += " Waiting..."
                    logging.info(info)
            else:
                logging.info("No GPU detected. Ensure a GPU is available.")
                break
            time.sleep(interval)  # type: ignore[arg-type]


class ModelAnalyzer:
    """Analyzes a given PyTorch model.

    Provides methods to calculate the FLOPs (Floating Point Operations),
    MACs (Multiply-Accumulate Operations), and total parameters of a given model,
    as well as to print a summary of the model.
    """

    @staticmethod
    def cal_flops(model: torch.nn.Module, input_shape: Tuple) -> Tuple:
        """Calculates the FLOPs, MACs, and total parameters of a given model."""
        from calflops import calculate_flops

        # pip install dynamic_network_architectures calflops
        return calculate_flops(model, input_shape=input_shape)

    @staticmethod
    def summarize(model: torch.nn.Module, input_size: Tuple, device: str = "cpu") -> None:
        """Prints a summary of the given model."""
        from torchsummary import summary

        summary(model, input_size=input_size, device=device)

    @staticmethod
    def test(
        model: Optional[torch.nn.Module] = None,
        img_size: tuple = (128, 128, 128),
        device: str = "cpu",
    ) -> None:
        """Tests the model by computing FLOPs and displaying its summary.

        If no model is provided, a default model is created.

        Args:
            model (Optional[torch.nn.Module]):
                Model to analyze (default=None).
            img_size (tuple):
                Size of the input image (default=(128, 128, 128)).
            device (str):
                Device on which to analyze the model (default="cpu").

        """
        from monai.networks.nets import SwinUNETR

        if model is None:
            model = SwinUNETR(img_size=img_size, in_channels=1, out_channels=1, feature_size=48)

        flops, macs, params = ModelAnalyzer.cal_flops(
            model,
            input_shape=(1, 1, *img_size),  # type: ignore[misc]
        )
        print(flops, macs, params)

        ModelAnalyzer.summarize(
            model,
            input_size=(1, *img_size),  # type: ignore[misc]
            device=device,  # type: ignore[arg-type]
        )
