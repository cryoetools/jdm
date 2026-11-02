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
"""Step 1."""

from __future__ import annotations

import argparse
import logging
import operator
import os
import sys
import warnings
from functools import partial, reduce
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from monai.inferers import sliding_window_inference
from omegaconf import OmegaConf

from cryojdm.utils import (
    DataHandler,
    FileHandler,
    FileManager,
    Rand,
    SplitTrainVal,
    StringUtils,
    path_resolve,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="\n%(asctime)s, %(levelname)-8s %(message)s\n",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)


load_mrc = FileHandler.load_mrc
save_mrc = FileHandler.save_mrc
save_json = FileHandler.save_json
mkdir_func = FileManager.create
sample_func = Rand.sample
norm_func = partial(DataHandler.scale_array, norm=True)
str2int = StringUtils.str2int
str2float = StringUtils.str2float
split_data = SplitTrainVal.split_data
split_path = SplitTrainVal.split_path


def create_parser() -> argparse.ArgumentParser:
    """Set up an argument parser for command-line input.

    Returns:
        argparse.ArgumentParser: Configured parser with expected arguments.
    """
    parser = argparse.ArgumentParser(description="Process command-line arguments.")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the pre-processing configuration file.",
    )
    return parser


class SlidingWindowSplitter(nn.Module):
    """A PyTorch module for performing sliding window splitting.

    Processes large input data in smaller chunks and saves patches to a specified path.

    Attributes:
        save_path (Union[Path, str]): Path to save patches.
        roi_size (Union[int, List[int]]): Region of interest size.
        overlap (float): Overlap between windows.
        padding_mode (Optional[str]): Padding mode (default="reflect").
        sw_batch_size (Optional[int]): Batch size for sliding window inference (default=1).
        voxel_size (Optional[float]): Voxel size for saving the patches (default=1.0).
    """

    def __init__(
        self,
        save_path: Union[Path, str],
        roi_size: Union[int, List[int]],
        overlap: float,
        padding_mode: Optional[str] = "reflect",
        sw_batch_size: Optional[int] = 1,
        voxel_size: Optional[float] = 1.0,
    ) -> None:
        """Initializes."""
        super().__init__()
        self.voxel_size = voxel_size
        self.save_path = path_resolve(save_path)
        self.save_path_list: List = []
        self.count = 0

        self.inference = partial(
            sliding_window_inference,
            predictor=self.predictor,
            roi_size=roi_size,
            overlap=overlap,
            padding_mode=padding_mode,  # type: ignore[arg-type]
            sw_batch_size=sw_batch_size,  # type: ignore[arg-type]
        )

    def save_patch(self, patches: list[torch.Tensor]) -> None:
        """Saves patches to the specified path."""
        for patch in patches:
            save_path = self.save_path.with_stem(f"{self.save_path.stem}_{self.count:05d}")
            save_mrc(save_path, patch.squeeze().numpy(), voxel_size=self.voxel_size)
            self.save_path_list.append(str(save_path))
            self.count += 1

    def predictor(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Saves patches and returns the input."""
        self.save_patch(x)
        return x

    def forward(self, x: torch.Tensor) -> list[str]:
        """Performs inference and returns saved paths."""
        x = self.inference(x)  # type: ignore[assignment]
        return self.save_path_list


class SplitPatch:
    """Splits input data into patches utilizes the `SlidingWindowSplitter`.

    Supports image binning and resizing.

    Attributes:
        patch_dir (Union[Path, str]): Directory to save patches.
        roi_size (Union[int, List[int]]): Region of interest size.
        overlap (float): Overlap between windows.
        padding_mode (Optional[str]): Padding mode (default="reflect").
        sw_batch_size (Optional[int]): Batch size for sliding window inference (default=1).
        voxel_size (Optional[float]): Voxel size for saving the patches (default=1.0).
    """

    def __init__(
        self,
        patch_dir: Union[Path, str],
        roi_size: Union[int, List[int]],
        overlap: float,
        padding_mode: Optional[str] = "reflect",
        sw_batch_size: Optional[int] = 1,
        voxel_size: Optional[float] = 1.0,
    ) -> None:
        """Initializes."""
        super().__init__()
        self.patch_dir = path_resolve(patch_dir)
        self.voxel_size = voxel_size
        self.splitter = partial(
            SlidingWindowSplitter,
            roi_size=roi_size,
            overlap=overlap,
            padding_mode=padding_mode,
            sw_batch_size=sw_batch_size,
        )
        self.resize_img_func = partial(F.interpolate, mode="area")
        self.resize_label_func = partial(F.interpolate, mode="nearest")

    def split(
        self,
        input_path: Path,
        label_flag: Optional[bool] = False,
        prefix: Optional[str] = "all",
        bin_factor: Optional[int] = 1,
    ) -> List[str]:
        """Splits data into patches and saves them."""
        save_path = self.patch_dir / f"{prefix}_bin{bin_factor}_{input_path.name}"
        splitter = self.splitter(
            save_path=save_path,
            voxel_size=self.voxel_size * bin_factor,  # type: ignore[operator]
        )

        input_data = (
            torch.from_numpy(norm_func(load_mrc(input_path))).unsqueeze(0).unsqueeze(0)  # type: ignore[arg-type]
        )
        shape = input_data.shape[2:]  # Extract spatial dimensions

        if bin_factor != 1:
            resize_func = self.resize_label_func if label_flag else self.resize_img_func
            input_data = resize_func(
                input_data.to(torch.float32),
                size=[s // bin_factor for s in shape],  # type: ignore[operator]
            )
            print(input_data.shape)
        return splitter(input_data)


class Deconv3D:
    """Perform 3D deconvolution on tomogram(s).

    Apply deconvolution to both individual tomogram (NumPy) and batches of tomograms (PyTorch).

    Attributes:
        tomo_shape (List[int]):
            The shape of the tomogram.
        angpix (Optional[float]):
            Angstroms per pixel (default=10.0).
        voltage (Optional[float]):
            Microscope voltage in kV (default=300.0).
        cs (Optional[float]):
            Spherical aberration in millimeters (default=2.7).
        defocus (Optional[float]):
            Defocus in angstroms, positive value means under focus (default=0.0).
        snrfalloff (Optional[float]):
            Rate of SNR falloff (default=1.0).
        deconvstrength (Optional[float]):
            How much will the signal be deconvoluted overall,
            i. e. a global scale for SNR;
            exponential scale: 1.0 is SNR = 1000 at zero frequency,
            0.67 is SNR = 100, and so on (default=1.0).
        highpassnyquist (Optional[float]):
            Fraction of Nyquist frequency for high-pass filtering (default=0.02).
        phaseflipped (Optional[bool]):
            Indicates if data is phase-flipped (default=False).
        phaseshift (Optional[float]):
            CTF phase shift in degrees (default=0).
        norm (Optional[bool]):
            Whether to normalize the output (default=False).

    Note:
        This class is derived from `tom_deconv.m` by Dimitri Tegunov.
        For more information, visit:
        https://github.com/dtegunov/tom_deconv/blob/master/tom_deconv.m
    """

    def __init__(
        self,
        tomo_shape: List[int],
        angpix: Optional[float] = 10.0,
        voltage: Optional[float] = 300.0,
        cs: Optional[float] = 2.7,
        defocus: Optional[float] = 0.0,
        snrfalloff: Optional[float] = 1.0,
        deconvstrength: Optional[float] = 1.0,
        highpassnyquist: Optional[float] = 0.02,
        phaseflipped: Optional[bool] = False,
        phaseshift: Optional[float] = 0,
        norm: Optional[bool] = False,
    ) -> None:
        """Initializes."""
        super().__init__()
        self.angpix = angpix
        self.voltage = voltage
        self.cs = cs
        self.defocus = defocus / 10000.0  # type: ignore[operator] # angstroms to micrometers
        self.snrfalloff = snrfalloff
        self.deconvstrength = deconvstrength
        self.highpassnyquist = highpassnyquist
        self.phaseflipped = phaseflipped
        self.phaseshift = phaseshift
        self.norm = norm

        self.amplitude = 0.07
        self.bfactor = 0
        self.ctf_length = 2048

        self.ramp_arr = self.get_ramp(tomo_shape)

    def tom_ctf1d(self) -> np.ndarray:
        """Compute the 1D Contrast Transfer Function (CTF) curve.

        Returns:
            np.ndarray: The computed CTF curve, adjusted by amplitude contrast and
            B-factor, as a 1D numpy array.

        Note:
            This method is a Python implementation of the MATLAB function `tom_ctf1d.m`
            by Dimitri Tegunov.
            For more information, visit:
            https://github.com/dtegunov/tom_deconv/blob/master/tom_ctf1d.m
        """
        pixelsize, voltage, cs, defocus, amplitude, phaseshift, bfactor, length = map(
            self.__getattribute__,
            (
                "angpix",
                "voltage",
                "cs",
                "defocus",
                "amplitude",
                "phaseshift",
                "bfactor",
                "ctf_length",
            ),
        )
        pixelsize = pixelsize * 1e-10
        voltage = voltage * 1e3
        cs = cs * 1e-3
        defocus = -defocus * 1e-6  # micrometers to meters
        phaseshift = phaseshift / 180 * np.pi

        ny = 1 / pixelsize

        lambda1 = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
        lambda2 = lambda1 * 2

        points = np.arange(0, length)
        points = points.astype(float)
        points = points / (2 * length) * ny

        k2 = points**2
        term1 = lambda1**3 * cs * k2**2

        # $\frac{\pi}{2}(\lambda^3 C_s k^4 + 2\lambda \Delta f k^2) - \text{phaseshift}$
        w = np.pi / 2 * (term1 + lambda2 * defocus * k2) - phaseshift

        acurve = np.cos(w) * amplitude
        pcurve = -np.sqrt(1 - amplitude**2) * np.sin(w)
        bfactor = np.exp(-bfactor * k2 * 0.25)

        return (pcurve + acurve) * bfactor

    def wiener1d(self) -> np.ndarray:
        """Compute the Wiener filter for 3D tomogram deconvolution.

        The filter is based on the 1D CTF curve.

        Returns:
            np.ndarray: The Wiener filter as a 1D array.
        """
        angpix, snrfalloff, deconvstrength, highpassnyquist, phaseflipped = map(
            self.__getattribute__,
            (
                "angpix",
                "snrfalloff",
                "deconvstrength",
                "highpassnyquist",
                "phaseflipped",
            ),
        )

        f = np.arange(0, 1 + 1 / 2047.0, 1 / 2047.0)
        highpass = np.minimum(np.ones_like(f), f / highpassnyquist) * np.pi
        highpass = 1 - np.cos(highpass)

        snr = (
            np.exp(-f * snrfalloff * 100 / angpix) * (10 ** (3 * deconvstrength)) * highpass + 1e-6
        )

        ctf = self.tom_ctf1d()
        if phaseflipped:
            ctf = abs(ctf)

        wiener = ctf / (ctf * ctf + 1 / snr)
        return wiener

    def get_ramp(self, tomo_shape: List[int]) -> np.ndarray:
        """Generate a 3D ramp filter for deconvolution based on the tomogram shape."""
        wiener = self.wiener1d()

        s1 = -int(tomo_shape[1] / 2)
        f1 = s1 + tomo_shape[1] - 1
        m1 = np.arange(s1, f1 + 1)

        s2 = -int(tomo_shape[0] / 2)
        f2 = s2 + tomo_shape[0] - 1
        m2 = np.arange(s2, f2 + 1)

        s3 = -int(tomo_shape[2] / 2)
        f3 = s3 + tomo_shape[2] - 1
        m3 = np.arange(s3, f3 + 1)

        x, y, z = np.meshgrid(m1, m2, m3)
        x = x.astype(np.float32) / np.maximum(1, np.abs(s1))
        y = y.astype(np.float32) / np.maximum(1, np.abs(s2))
        z = z.astype(np.float32) / np.maximum(1, np.abs(s3))

        # Compute radial distance and apply threshold
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.minimum(1, r)
        r = np.fft.ifftshift(r)

        # Interpolate to generate the ramp
        f = np.arange(0, 1 + 1 / 2047.0, 1 / 2047.0)
        ramp = np.interp(r, f, wiener).astype(np.float32)

        return ramp

    def deconv_array(
        self,
        tomo: np.ndarray,
        ncpu: Optional[int] = os.cpu_count() - 5,  # type: ignore[operator]
    ) -> np.ndarray:
        """Apply deconvolution to a single tomogram."""
        ramp = self.ramp_arr
        std_raw = np.std(tomo)
        ave_raw = np.average(tomo)

        # Perform Fourier transform, apply ramp filter, and inverse Fourier transform
        tomo = np.real(
            scipy.fft.ifftn(
                scipy.fft.fftn(tomo, overwrite_x=True, workers=ncpu) * ramp,
                overwrite_x=True,
                workers=ncpu,
            )
        )
        tomo = tomo.astype(np.float32)

        if self.norm:
            std_now = np.std(tomo)
            tomo /= std_now
            tomo *= std_raw
            tomo += ave_raw
        return tomo

    def deconv_batch(self, tomo_batch: torch.Tensor) -> torch.Tensor:
        """Apply 3D deconvolution to a batch of tomograms."""
        dim = tuple(range(2, tomo_batch.ndim))

        std_raw = tomo_batch.std(dim=dim, keepdim=True)
        ave_raw = tomo_batch.mean(dim=dim, keepdim=True)

        device = tomo_batch.device
        ramp = torch.from_numpy(self.ramp_arr[None, None, ...]).to(device)
        ramp = ramp.repeat(tomo_batch.shape[0], 1, 1, 1, 1)

        tomo_batch = torch.fft.fftn(tomo_batch, dim=dim, norm="ortho") * ramp
        tomo_batch = (torch.fft.ifftn(tomo_batch, dim=dim, norm="ortho")).real

        if self.norm:
            std_now = tomo_batch.std(dim=dim, keepdim=True)
            tomo_batch /= std_now
            tomo_batch *= std_raw
            tomo_batch += ave_raw
        return tomo_batch

    def __call__(self, tomo: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Execute deconvolution on the input tomogram(s)."""
        tomo = (
            self.deconv_batch(tomo) if isinstance(tomo, torch.Tensor) else self.deconv_array(tomo)
        )
        return tomo


def deconv_paths(
    output_dir: Path,
    input_paths: List[Path],
    snrfalloff: float,
    deconvstrength: float,
    defocus: float,
    angpix: float,
    img_invert: Optional[bool] = False,
) -> List[Path]:
    """Applies 3D deconvolution to MRC files and saves results.

    Supports image deconvolution, normalization, and inversion.

    Args:
        output_dir (Path): Directory to save deconvolved files.
        input_paths (List[Path]): Paths to input MRC files.
        snrfalloff (float): SNR falloff for deconvolution.
        deconvstrength (float): Deconvolution strength.
        defocus (float): Defocus value.
        angpix (float): Angstrom per pixel.
        img_invert (Optional[bool]): Invert image after deconvolution if True (default=False).

    Returns:
        List[Path]: Paths to saved deconvolved files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for p in input_paths:
        output_path = output_dir / f"{p.stem}.mrc"
        output_paths.append(output_path)

        if output_path.exists():
            continue

        tomo = load_mrc(p)
        deconv_func = Deconv3D(
            tomo_shape=list(tomo.shape),  # type: ignore[union-attr]
            snrfalloff=snrfalloff,
            deconvstrength=deconvstrength,
            defocus=defocus,
            angpix=angpix,
        )

        tomo = deconv_func(tomo)  # type: ignore[arg-type]
        tomo = norm_func(-tomo if img_invert else tomo)  # type: ignore[arg-type]
        save_mrc(output_path, tomo, voxel_size=angpix)

    return output_paths


def launch_pre() -> None:  # noqa: C901
    """Initializes and runs."""
    logging.info("Start!")

    # ==========
    # Load and adjust configuration settings

    args = create_parser().parse_args()
    cfgs = OmegaConf.load(args.config_file)

    # Extract configuration values
    (
        mode,
        base_dir,
        seed,
        bin_factor,
        voxel_size,
        val_setup,
        roi_len,
        overlap,
        deconv_apply,
        defocus,
        snrfalloff,
        deconvstrength,
    ) = map(
        cfgs.get,  # type: ignore[arg-type]
        (
            "mode",
            "base_dir",
            "seed",
            "bin_factor",
            "voxel_size",
            "val_setup",
            "roi_len",
            "overlap",
            "apply_deconv",
            "defocus",
            "snrfalloff",
            "deconvstrength",
        ),
    )

    # Convert specific configuration values
    val_setup, snrfalloff, deconvstrength = map(str2float, [val_setup, snrfalloff, deconvstrength])
    bin_factor = str2int(bin_factor)

    # Resolve base directory paths
    base_dir = path_resolve(base_dir)
    input_dir = base_dir / "0_input" / "data"
    output_dir = base_dir / "1_preprocess"

    # Create necessary output directories
    out_json_dir = output_dir / "json"
    out_data_dir = output_dir / "data"
    out_patch_dir = out_data_dir / "split" / "patch"
    out_noisy_dir = (
        out_data_dir / "noisy" / "deconv" if deconv_apply else out_data_dir / "noisy" / "norm"
    )
    [mkdir_func(d) for d in [out_json_dir, out_noisy_dir, out_patch_dir]]

    # Determine input directories
    in_label_dir = input_dir / "label"
    in_clean_dir = input_dir / "clean"
    in_all_dir = input_dir / "noisy" / "all"
    in_all_dir = in_all_dir if in_all_dir.exists() else input_dir / "noisy"
    in_odd_dir = input_dir / "noisy" / "odd"
    in_odd_dir = in_odd_dir if in_odd_dir.exists() else None
    in_even_dir = input_dir / "noisy" / "even"
    in_even_dir = in_even_dir if in_even_dir.exists() else None

    # Adjust processing mode based on input directories
    if mode != 0 and None in [in_odd_dir, in_even_dir]:
        mode = 0

    # Initialize parameters for data patch processing
    suffix = "mrc"
    sw_batch_size = os.cpu_count() - 5  # type: ignore[operator]
    spatial_dims = 3
    split_roi_len = np.ceil(roi_len * np.sqrt(2))
    split_roi_len += split_roi_len % 2
    split_roi_size = [int(split_roi_len)] * spatial_dims
    sp_func = SplitPatch(
        patch_dir=out_patch_dir,
        roi_size=split_roi_size,
        overlap=overlap,
        sw_batch_size=sw_batch_size,
        voxel_size=voxel_size,
    )

    # ==========
    # Process `path_dict` containing 'clean' and 'label' files.

    path_dict = {}  # type: ignore[var-annotated]

    if in_clean_dir.exists():
        path_dict.update({"clean": []})
        [path_dict["clean"].append(p) for p in sorted(in_clean_dir.rglob(f"*.{suffix}"))]

    if in_label_dir.exists():
        path_dict.update({"label": []})
        [path_dict["label"].append(p) for p in sorted(in_label_dir.rglob(f"*.{suffix}"))]

    # ==========
    # Process `path_dict` containing 'noisy' files based on the mode and
    # apply deconvolution or normalization.

    if mode == 0:
        raw_dirs = [in_all_dir]
        folders = ["all"]
    else:
        raw_dirs = [in_all_dir, in_odd_dir, in_even_dir]
        folders = ["all", "odd", "even"]

    path_dict.update({k: [] for k in folders})
    for raw_dir, folder in zip(raw_dirs, folders, strict=False):
        folder_dir = out_noisy_dir / f"{folder}"

        if deconv_apply:
            # Check if there's only one set of deconvolution parameters
            if len(snrfalloff) == 1:
                s, d = snrfalloff[0], deconvstrength[0]
                out_dir = out_noisy_dir / f"{folder}_s{s:02.2f}_d{d:02.2f}"
                _ = deconv_paths(
                    output_dir=out_dir,
                    input_paths=sorted(raw_dir.rglob(f"*.{suffix}")),
                    snrfalloff=s,
                    deconvstrength=d,
                    defocus=defocus,
                    angpix=voxel_size,
                )
                if folder_dir.exists():
                    os.system(f"rm -rf {folder_dir}")  # noqa: S605
                os.system(f"ln -s {out_dir} {folder_dir}")  # noqa: S605

            else:
                # Calculate the mean of the deconvolved results
                mean_dir = out_noisy_dir / f"{folder}_mean"
                mkdir_func(mean_dir)

                snrfalloffs = np.arange(snrfalloff[0], snrfalloff[1], 0.5)
                deconvstrengths = np.arange(deconvstrength[0], deconvstrength[1], 0.5)
                output_paths_list = []
                for s, d in zip(snrfalloffs, deconvstrengths, strict=False):
                    output_paths = deconv_paths(
                        output_dir=out_noisy_dir / f"{folder}_s{s:02.2f}_d{d:02.2f}",
                        input_paths=sorted(raw_dir.rglob(f"*.{suffix}")),
                        snrfalloff=s,
                        deconvstrength=d,
                        defocus=defocus,
                        angpix=voxel_size,
                    )
                    output_paths_list.append(output_paths)

                for i in range(len(output_paths_list[0])):
                    paths = [sublist[i] for sublist in output_paths_list]
                    stem = paths[0].stem
                    mean_path = mean_dir / f"{stem}.mrc"

                    if mean_path.exists():
                        continue

                    print(f"Calculating mean for: {paths}")
                    tomos = [load_mrc(p) for p in paths]
                    tomos = norm_func(np.mean(np.stack(tomos), axis=0))  # type: ignore[assignment]
                    save_mrc(mean_path, tomos, voxel_size=voxel_size)  # type: ignore[arg-type]

                if folder_dir.exists():
                    os.system(f"rm -rf {folder_dir}")  # noqa: S605
                os.system(f"ln -s {mean_dir} {folder_dir}")  # noqa: S605

        else:
            # Normalize
            mkdir_func(folder_dir)
            for p in sorted(raw_dir.rglob(f"*.{suffix}")):
                save_path = folder_dir / p.name
                if save_path.exists():
                    continue
                save_mrc(save_path, norm_func(load_mrc(p)), voxel_size=voxel_size)  # type: ignore[arg-type]

        path_dict[folder] = sorted(folder_dir.glob("*.mrc"))

    print(f"path_dict: {path_dict}")

    # ==========
    # Split files into validation and training sets based on a specified ratio or indices.

    val_path_dict = {k: [] for k in path_dict}  # type: ignore[var-annotated]
    train_path_dict = {k: [] for k in path_dict}  # type: ignore[var-annotated]

    if 0.1 <= val_setup[0] <= 1:
        val_ratio = float(val_setup[0])
        val_dir = out_data_dir / "split" / "val"
        train_dir = out_data_dir / "split" / "train"
        for k, paths in path_dict.items():
            for p in paths:
                val_path = val_dir / f"{k}" / p.name
                train_path = train_dir / f"{k}" / p.name
                val_path_dict[k].append(val_path)
                train_path_dict[k].append(train_path)

                if train_path.exists():
                    continue

                val_arr, train_arr = split_data(load_mrc(p), val_ratio)  # type: ignore[arg-type]
                [mkdir_func(path.parent) for path in [val_path, train_path]]
                save_mrc(val_path, val_arr, voxel_size=voxel_size)
                save_mrc(train_path, train_arr, voxel_size=voxel_size)

    else:
        val_indices = [int(i) for i in val_setup]
        for k, paths in path_dict.items():
            val_path, train_path = split_path(paths, val_indices)
            val_path_dict[k] = val_path
            train_path_dict[k] = train_path

    print(f"val_path_dict: {val_path_dict}")

    # ==========
    # Split training and validation data into patches and organize them into dictionaries.
    # The training data using all specified bin factors,
    # while the validation data using a bin factor of 1.

    # train
    train_patch_dict = {k: [] for k in path_dict}  # type: ignore[var-annotated]
    for b in bin_factor:
        for k, paths in train_path_dict.items():
            for p in paths:
                train_patch_paths = sp_func.split(
                    input_path=p,
                    label_flag=(k == "label"),
                    prefix=k,
                    bin_factor=b,
                )
                train_patch_dict[k].append(train_patch_paths)

    # val
    val_patch_dict = {k: [] for k in path_dict}  # type: ignore[var-annotated]
    for b in [1]:
        for k, paths in val_path_dict.items():
            for p in paths:
                val_patch_paths = sp_func.split(
                    input_path=p,
                    label_flag=(k == "label"),
                    prefix=k,
                    bin_factor=b,
                )
                val_patch_dict[k].append(val_patch_paths)

    # ==========
    # Generate a dataset dictionary containing paths for training,
    # validation, and inference datasets. Save the dictionary as a JSON file.

    dataset_dict = {"train": [], "val": [], "infer": []}  # type: ignore[var-annotated]

    # train
    [
        train_patch_dict.update(
            {k: sample_func(reduce(operator.iadd, train_patch_dict[k], []), seed=seed)}  # type: ignore[dict-item]
        )
        for k in train_patch_dict
    ]
    keys = ["clean", "label", "all", "odd", "even"]
    for i in range(len(train_patch_dict["all"])):
        set_dict = {"nth": i}
        set_dict.update(
            {
                k: str(path_resolve(train_patch_dict[k][i]))  # type: ignore[misc]
                for k in keys
                if k in train_patch_dict
            }
        )
        dataset_dict["train"].append(set_dict)

    # val
    [
        val_patch_dict.update({k: reduce(operator.iadd, val_patch_dict[k], [])})
        for k in val_patch_dict
    ]
    keys = ["clean", "label", "all", "odd", "even"]
    for i in range(len(val_patch_dict["all"])):
        set_dict = {"nth": i}
        set_dict.update(
            {
                k: str(path_resolve(val_patch_dict[k][i]))  # type: ignore[misc]
                for k in keys
                if k in val_patch_dict
            }
        )
        dataset_dict["val"].append(set_dict)

    # infer
    keys = ["clean", "label", "all"]
    for i in range(len(path_dict["all"])):
        set_dict = {
            k: str(path_resolve(path_dict[k][i]))  # type: ignore[misc]
            for k in keys
            if k in path_dict
        }
        dataset_dict["infer"].append(set_dict)

    # Save dataset dictionary to JSON
    save_json(out_json_dir / "train.json", dataset_dict)
    logging.info("Finish!")


if __name__ == "__main__":
    launch_pre()
