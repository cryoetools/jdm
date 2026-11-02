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
"""GUI application."""

import logging
import os
import signal
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.messagebox as messagebox
from functools import partial
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Dict, Tuple, Union

import customtkinter
import mrcfile
import numpy as np
import yaml  # type: ignore[import-untyped]
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from .. import __version__

CORE_PATH = Path(__file__).resolve().parent.parent
RESOURCES_PATH = Path(__file__).resolve().parent / "resources"
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme(RESOURCES_PATH / "color.json")
version = f"v{__version__}"

run_python = sys.executable
step_1 = str(CORE_PATH / "preprocess.py")
step_2 = str(CORE_PATH / "main.py")
step_3 = str(CORE_PATH / "postprocess.py")


class Converter:
    """Converts input between different formats.

    Provides methods to convert data to YAML format,
    and to parse strings into integers or lists of integers.
    """

    @staticmethod
    def to_yaml(inp: Union[ListConfig, DictConfig]) -> str:
        """Converts the input to a YAML formatted string."""
        if not isinstance(inp, DictConfig):
            inp = OmegaConf.create(inp)
        container = OmegaConf.to_container(inp, resolve=True)
        return yaml.dump(container, default_flow_style=None, sort_keys=False)

    @staticmethod
    def to_int_list(inp: str) -> Union[int, list]:
        """Converts a string to an integer or a list of integers."""
        import ast

        inp = inp.strip()
        if "[" in inp and "]" in inp:
            return ast.literal_eval(inp)
        elif "," in inp:
            return [int(x) for x in inp.strip("[]").split(",")]
        else:
            return int(inp)


class Processor:
    """Processes MRC files and applies various image transformations."""

    @staticmethod
    def load_mrc(load_path: Union[str, Path]) -> np.ndarray:
        """Loads data from an MRC file."""
        with mrcfile.open(load_path, permissive=True) as mrc:
            data = mrc.data
        return data

    @staticmethod
    def get_plane(arr: np.ndarray, idx: int, plane: str = "xy") -> np.ndarray:
        """Extracts a 2D slice from a 3D numpy array based on the specified plane and index."""
        if plane == "xy":
            return arr[idx]
        elif plane == "xz":
            return arr[:, idx, :]
        elif plane == "yz":
            return np.rot90(arr[:, :, idx])
        else:
            raise ValueError("Invalid plane specified. Choose from 'xy', 'xz', or 'yz'.")

    @staticmethod
    def get_scaled(arr: np.ndarray) -> np.ndarray:
        """Scales the input array to the range [0, 1]."""
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-6)

    @staticmethod
    def to_int(arr: np.ndarray) -> np.ndarray:
        """Scales array values by 255, clips to [0, 255], and converts to uint8."""
        return np.clip(arr * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def to_ctkimg(arr: np.ndarray) -> "customtkinter.CTkImage":
        """Converts a 2D numpy array to a `customtkinter` image object."""
        img = Image.fromarray(arr, mode="L")
        return customtkinter.CTkImage(img, size=arr.shape[::-1])

    @staticmethod
    def get_rescaled(arr: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
        """Rescales the intensity of the input array using a given percentile range."""
        from skimage import exposure

        arr = Processor.get_scaled(arr)
        p_low, p_high = np.percentile(arr, (low, high))
        processed = exposure.rescale_intensity(arr, in_range=(p_low, p_high))
        return Processor.get_scaled(processed)

    @staticmethod
    def get_inverted(arr: np.ndarray) -> np.ndarray:
        """Inverts the input array and scales it to the range [0, 1]."""
        return Processor.get_scaled(-arr)

    @staticmethod
    def get_fft(arr: np.ndarray, log_theta: float = 2) -> np.ndarray:
        """Computes the FFT of the input array and returns its log-scaled magnitude."""
        arr = np.fft.fftshift(np.fft.fftn(arr))
        arr = np.log(np.abs(arr) + log_theta)
        return Processor.get_scaled(arr)


read_mrc = Processor.load_mrc
get_xy = partial(Processor.get_plane, plane="xy")
get_xz = partial(Processor.get_plane, plane="xz")
get_yz = partial(Processor.get_plane, plane="yz")
get_scaled = Processor.get_scaled
get_intensity_rescaled = Processor.get_rescaled
get_inverted = Processor.get_inverted
get_fft = Processor.get_fft
to_int = Processor.to_int
to_ctkimg = Processor.to_ctkimg


class CustomInputDialog(customtkinter.CTkInputDialog):
    """Customs input dialog with customizable dimensions."""

    def __init__(
        self,
        title: str = "Custom Dialog",
        text: str = "Please input:",
        width: int = 300,
        height: int = 150,
    ) -> None:
        """Initializes."""
        super().__init__(title=title, text=text)
        self.geometry(f"{width}x{height}")


class TextboxHandler(logging.Handler):
    """Customs logging handler to direct log messages to a GUI textbox widget."""

    def __init__(self, textbox: customtkinter.CTkTextbox) -> None:
        """Initializes."""
        super().__init__()
        self.textbox = textbox

    def emit(self, record: logging.LogRecord) -> None:
        """Inserts log messages into the textbox."""
        log_message = self.format(record)
        self.textbox.insert("end", log_message + "\n")
        self.textbox.see("end")


def setup_logging(textbox: customtkinter.CTkTextbox, logger_name: str) -> logging.Logger:
    """Configures a logger to use the TextboxHandler with a specified format."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    textbox_handler = TextboxHandler(textbox)
    formatter = logging.Formatter(
        fmt="%(asctime)s, %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    textbox_handler.setFormatter(formatter)
    logger.addHandler(textbox_handler)
    return logger


class App(customtkinter.CTk):
    """Represents the main application window using customtkinter."""

    def __init__(self, width: int = 1100, height: int = 580, version: str = version) -> None:
        """Initializes."""
        super().__init__()

        self.title(version)
        self.geometry(f"{width}x{height}")

        # Set grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Load logo and configs
        self.logo_image = customtkinter.CTkImage(
            Image.open(RESOURCES_PATH / "logo.png"), size=(200, 100)
        )
        self.cfg_model = OmegaConf.load(RESOURCES_PATH / "default_model.yaml")

        # To keep track of the running process
        self.process = None
        self.stop_event = threading.Event()
        self.loggers: Dict[str, logging.Logger] = {}

        # Create panel
        self.build_navigation()
        self.build_first_panel()
        self.build_second_panel()
        self.build_third_panel()
        self.build_fourth_panel()

        # Select default panel
        self.select_panel_event("first_panel")
        self.appearance_mode_menu.set("System")
        self.scaling_optionemenu.set("100%")

    # ===== basic =====

    def run_cmd(self, textbox: customtkinter.CTkInputDialog, cmd: str) -> None:
        """Executes command in a separate thread and appends its output to textbox."""
        self.stop_event.clear()  # Reset the stop event

        def append_text(textbox: customtkinter.CTkInputDialog, text: str) -> None:
            """Appends text to the textbox and scrolls to the end."""
            textbox.insert("end", text + "\n")
            textbox.yview_moveto(1)

        def execute() -> None:
            try:
                self.process = subprocess.Popen(  # noqa: S602
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                    text=True,
                    preexec_fn=os.setsid,  # type: ignore[attr-defined]
                )

                while True:
                    if self.stop_event.is_set():
                        # Terminate the process group
                        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)  # type: ignore[attr-defined]
                        append_text(textbox, "Command terminated by user.")
                        break

                    # Read output line by line
                    if self.process.stdout is not None:
                        line = self.process.stdout.readline()
                        if not line:  # Break loop when no more output
                            break
                        append_text(textbox, line.strip())

                self.process.wait()  # Wait for process to exit
                if not self.stop_event.is_set():  # Only show finished message if not terminated
                    append_text(
                        textbox,
                        f"Command finished with return code {self.process.returncode}",
                    )
            except Exception as e:  # noqa: BLE001
                append_text(textbox, f"Error: {e!s}")
            finally:
                # Reset buttons after command finishes
                self.process = None

        threading.Thread(target=execute, daemon=True).start()

    def get_label(
        self,
        master: customtkinter,
        text: str,
        row: int,
        column: int,
        pady: Union[int, Tuple] = 5,
    ) -> None:
        """Creates and places a custom label widget in the specified grid location."""
        label = customtkinter.CTkLabel(master, text=text, anchor="e", compound="right")
        label.grid(row=row, column=column, padx=(20, 0), pady=pady, sticky="e")

    def create_control_button(
        self,
        frame: customtkinter.CTkFrame,
        text: str,
        command: Callable,
        size: int = 14,
        weight: str = "normal",
        width: int = 140,
    ) -> customtkinter.CTkButton:
        """Creates a custom button with specified properties."""
        return customtkinter.CTkButton(
            frame,
            text=text,
            command=command,
            font=customtkinter.CTkFont(size=size, weight=weight),
            width=width,
        )

    # ================================

    def build_navigation(self) -> None:
        """Constructs the navigation panel."""

        def create_button(
            master: customtkinter.CTkFrame,
            text: str,
            command: Callable,
            image: customtkinter.CTkImage = None,
        ) -> customtkinter.CTkButton:
            """Creates a custom button with specified properties."""
            return customtkinter.CTkButton(
                master,
                corner_radius=0,
                height=40,
                border_spacing=20,
                text=text,
                font=customtkinter.CTkFont(size=16, weight="bold"),
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                image=image,
                anchor="w",
                command=command,
            )

        buttons_info = [
            ("Preprocess", "first_panel"),
            ("Train", "second_panel"),
            ("Infer", "third_panel"),
            ("Display", "fourth_panel"),
        ]
        buttons_num = len(buttons_info)

        navigation_panel = customtkinter.CTkFrame(self, corner_radius=0)
        master = navigation_panel
        master.grid(row=0, column=0, sticky="nsew")
        master.grid_rowconfigure(buttons_num + 1, weight=1)
        padx = 20
        pady = 20

        # LOGO
        logo = customtkinter.CTkLabel(
            master,
            text="",
            image=self.logo_image,
            compound="center",
        )
        logo.grid(row=0, column=0, padx=padx, pady=pady)

        # Buttons
        for idx, (text, panel_name) in enumerate(buttons_info, start=1):
            button = create_button(
                master,
                text=text,
                command=partial(self.select_panel_event, panel_name),
            )
            button.grid(row=idx, column=0, sticky="ew")
            setattr(self, f"{panel_name}_button", button)

        # Adjust frame
        adjust_frame = customtkinter.CTkFrame(master, fg_color="transparent")
        adjust_frame.grid(row=buttons_num + 2, column=0, padx=padx, pady=pady, sticky="nsew")

        ## Appearance Mode
        self.appearance_mode_menu = customtkinter.CTkOptionMenu(
            adjust_frame,
            width=90,
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode_event,
            anchor="center",
        )
        self.appearance_mode_menu.grid(row=0, column=0, padx=(0, 10), pady=0)

        ## UI Scaling
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(
            adjust_frame,
            width=90,
            values=["80%", "100%", "120%", "140%", "160%", "180%", "200%"],
            command=self.change_scaling_event,
            anchor="center",
        )
        self.scaling_optionemenu.grid(row=0, column=1, padx=(10, 0), pady=0)

    def build_first_panel(self) -> None:
        """Constructs the first panel."""
        self.pre_var: Dict = {}

        self.first_panel = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        master = self.first_panel
        master.grid_columnconfigure((0), weight=1)
        master.grid_rowconfigure((1), weight=1)

        # ----- control_frame -----

        control_frame = customtkinter.CTkFrame(master)
        control_frame.grid(row=0, column=1, rowspan=2, padx=0, pady=0, sticky="nsew")
        control_frame.grid_columnconfigure(0, weight=1)

        fc_1 = customtkinter.CTkFrame(control_frame, fg_color="transparent")
        fc_1.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        fc_2 = customtkinter.CTkFrame(control_frame, fg_color="transparent")
        fc_2.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        fc_2.grid_columnconfigure((0, 1, 2), weight=1)
        fc_3 = customtkinter.CTkFrame(control_frame, fg_color="transparent")
        fc_3.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="nsew")

        buttons_info = [
            (fc_1, "Load", self.load_pre_config_event, 14, "bold", 140, 0, 0),
            (fc_2, "Clear", self.clear_pre_event, 12, "normal", 20, 0, 0),
            (fc_2, "Save", self.save_pre_event, 12, "normal", 20, 1, 5),
            (fc_2, "Stop", self.stop_cmd_event, 12, "normal", 20, 2, 0),
            (fc_3, "Run", self.run_pre_event, 14, "bold", 140, 0, 0),
        ]

        # Buttons
        for frame, text, command, size, weight, width, column, padx in buttons_info:
            button = self.create_control_button(frame, text, command, size, weight, width)
            button.grid(row=0, column=column, padx=padx, pady=0, sticky="nsew")

        # ----- frame_2, scroll_1 -----

        padx = (10, 20)  # type: ignore[assignment]
        pady = 5

        frame_2 = customtkinter.CTkFrame(master, fg_color="transparent")
        frame_2.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="nsew")
        frame_2.grid_columnconfigure((0, 1), weight=1)
        frame_2.grid_rowconfigure(0, weight=1)

        scroll_1 = customtkinter.CTkScrollableFrame(frame_2, label_text=" ", height=240)
        scroll_1.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")
        scroll_1.grid_columnconfigure(1, weight=1)

        infos = [
            (
                "Base dir:",
                "base_dir",
                customtkinter.CTkOptionMenu,
                {
                    "command": self.select_pre_base_event,
                    "values": ["Click to select..."],
                },
            ),
            (
                "Val setup:",
                "val_setup",
                customtkinter.CTkButton,
                {"command": self.val_set_event, "anchor": "w", "compound": "left"},
            ),
            ("Mode:", "mode", customtkinter.CTkComboBox, {"values": ["0", "1", "2"]}),
            ("Seed:", "seed", customtkinter.CTkEntry, {}),
            ("Bin factor:", "bin_factor", customtkinter.CTkEntry, {}),
            ("Voxel size:", "voxel_size", customtkinter.CTkEntry, {}),
        ]

        for row, (label_text, var_name, widget_class, widget_kwargs) in enumerate(infos):
            self.get_label(
                scroll_1,
                text=label_text,
                row=row,
                column=0,
                pady=(0, pady) if row == 0 else pady,
            )
            self.pre_var[var_name] = tk.StringVar()
            if widget_class in [
                customtkinter.CTkOptionMenu,
                customtkinter.CTkComboBox,
                customtkinter.CTkCheckBox,
            ]:
                widget = widget_class(scroll_1, variable=self.pre_var[var_name], **widget_kwargs)
            else:
                widget = widget_class(
                    scroll_1, textvariable=self.pre_var[var_name], **widget_kwargs
                )
            widget.grid(
                row=row,
                column=1,
                padx=padx,
                pady=(0, pady) if row == 0 else pady,
                sticky="ew",
            )

        # ----- frame_2, scroll_2 -----

        scroll_2 = customtkinter.CTkScrollableFrame(frame_2, label_text=" ", height=240)
        scroll_2.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="nsew")
        scroll_2.grid_columnconfigure(1, weight=1)

        infos = [
            ("RoI length:", "roi_len", customtkinter.CTkEntry, {}),
            ("Overlap:", "overlap", customtkinter.CTkEntry, {}),
            ("Apply deconv:", "apply_deconv", customtkinter.CTkCheckBox, {"text": ""}),
            ("Defocus:", "defocus", customtkinter.CTkEntry, {}),
            ("Snrfalloff:", "snrfalloff", customtkinter.CTkEntry, {}),
            ("Deconvstrength:", "deconvstrength", customtkinter.CTkEntry, {}),
        ]

        for row, (label_text, var_name, widget_class, widget_kwargs) in enumerate(infos):
            self.get_label(
                scroll_2,
                text=label_text,
                row=row,
                column=0,
                pady=(0, pady) if row == 0 else pady,
            )

            if var_name == "apply_deconv":
                self.pre_var[var_name] = tk.IntVar()  # type: ignore[assignment]
            else:
                self.pre_var[var_name] = tk.StringVar()

            if widget_class in [
                customtkinter.CTkOptionMenu,
                customtkinter.CTkComboBox,
                customtkinter.CTkCheckBox,
            ]:
                widget = widget_class(scroll_2, variable=self.pre_var[var_name], **widget_kwargs)
            else:
                widget = widget_class(
                    scroll_2, textvariable=self.pre_var[var_name], **widget_kwargs
                )
            widget.grid(
                row=row,
                column=1,
                padx=padx,
                pady=(0, pady) if row == 0 else pady,
                sticky="ew",
            )

        # create textbox
        self.pre_textbox = customtkinter.CTkTextbox(master, width=200, wrap="word")
        self.pre_textbox.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")  # , rowspan=2
        logger_name = "pre"
        self.loggers[logger_name] = setup_logging(self.pre_textbox, logger_name)

    def get_tabtext(self, model_name: str = "ours") -> None:
        """Creates a new tab for the specified model and initializes its configuration."""
        self.model_tabview.add(model_name)
        mtab = self.model_tabview.tab(model_name)
        mtab.grid_columnconfigure(0, weight=1)
        mtab.grid_rowconfigure(0, weight=1)

        # Initialize the text box for the model's configuration
        self.train_tabtext[model_name] = customtkinter.CTkTextbox(
            mtab,
            wrap="word",
            fg_color="transparent",
            height=40,
        )
        self.train_cfg[model_name] = tk.StringVar()
        self.train_cfg[model_name].set(Converter.to_yaml(self.cfg_model[model_name]))  # type: ignore[index]

        # Configure the text box layout and insert the configuration
        self.train_tabtext[model_name].grid(row=0, column=0, sticky="nsew")
        self.train_tabtext[model_name].insert("end", self.train_cfg[model_name].get())

    def build_second_panel(self) -> None:
        """Constructs the second panel."""
        self.train_var: Dict = {}
        self.train_tabtext: Dict = {}
        self.train_cfg: Dict = {}

        self.second_panel = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        master = self.second_panel
        master.grid_columnconfigure((0), weight=1)
        master.grid_rowconfigure((0, 1), weight=1)

        # ----- control_frame -----

        control_frame = customtkinter.CTkFrame(master)
        control_frame.grid(row=0, column=1, rowspan=2, padx=0, pady=0, sticky="nsew")
        control_frame.grid_columnconfigure(0, weight=1)

        fc_1 = customtkinter.CTkFrame(control_frame, fg_color="transparent")
        fc_1.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        fc_2 = customtkinter.CTkFrame(control_frame, fg_color="transparent")
        fc_2.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        fc_2.grid_columnconfigure((0, 1, 2), weight=1)
        fc_3 = customtkinter.CTkFrame(control_frame, fg_color="transparent")
        fc_3.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        fc_4 = customtkinter.CTkFrame(control_frame, fg_color="transparent")
        fc_4.grid(row=3, column=0, padx=20, pady=(10, 20), sticky="nsew")

        buttons_info = [
            (fc_1, "Load", self.load_train_config_event, 14, "bold", 140, 0, 0),
            (fc_2, "Clear", self.clear_train_event, 12, "normal", 20, 0, 0),
            (fc_2, "Save", self.save_train_event, 12, "normal", 20, 1, 5),
            (fc_2, "Stop", self.stop_cmd_event, 12, "normal", 20, 2, 0),
            (fc_3, "Run", self.run_train_event, 14, "bold", 140, 0, 0),
        ]

        # Buttons
        for frame, text, command, size, weight, width, column, padx in buttons_info:
            button = self.create_control_button(frame, text, command, size, weight, width)
            button.grid(row=0, column=column, padx=padx, pady=0, sticky="nsew")

        self.train_var["memo"] = tk.StringVar()
        train_memo_entry = customtkinter.CTkEntry(fc_4, textvariable=self.train_var["memo"])
        train_memo_entry.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        # ----- frame_2, scroll_1 -----

        padx = (10, 20)  # type: ignore[assignment]
        pady = 5

        frame_2 = customtkinter.CTkFrame(master, fg_color="transparent")
        frame_2.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="nsew")
        frame_2.grid_columnconfigure((0, 1), weight=1)
        frame_2.grid_rowconfigure(0, weight=1)

        scroll_1 = customtkinter.CTkScrollableFrame(frame_2, label_text=" ")
        scroll_1.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")
        scroll_1.grid_columnconfigure(0, weight=1)

        s1_1 = customtkinter.CTkFrame(scroll_1)
        s1_1.grid(row=0, column=0, padx=0, pady=(0, pady), sticky="nsew")
        s1_1.grid_columnconfigure(2, weight=1)
        self.get_label(s1_1, text="Continue train:", row=0, column=0, pady=0)

        self.pretrain_check = customtkinter.CTkCheckBox(
            s1_1, width=24, text=":", command=self.pretrain_event
        )
        self.pretrain_check.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="ew")
        self.train_var["pretrained_path"] = tk.StringVar()
        pretrain_entry = customtkinter.CTkEntry(
            s1_1, textvariable=self.train_var["pretrained_path"]
        )
        pretrain_entry.grid(row=0, column=2, padx=padx, pady=0, sticky="ew")

        s1_2 = customtkinter.CTkFrame(scroll_1)
        s1_2.grid(row=1, column=0, padx=0, pady=(0, pady), sticky="nsew")
        s1_2.grid_columnconfigure(1, weight=1)

        infos = [
            (
                "Base dir:",
                "base_dir",
                customtkinter.CTkOptionMenu,
                {
                    "command": self.select_train_base_event,
                    "values": ["Click to select..."],
                },
            ),
            (
                "Json path:",
                "json_path",
                customtkinter.CTkOptionMenu,
                {
                    "command": self.select_train_json_event,
                    "values": ["Click to select..."],
                },
            ),
            (
                "Model:",
                "model",
                customtkinter.CTkOptionMenu,
                {
                    "command": self.select_model_event,
                    "values": [
                        "ours",
                        "emnet",
                        "umamba",
                        "segresnet",
                        "unet",
                        "unetr",
                        "swinunetr",
                    ],
                },
            ),
            ("Mode:", "mode", customtkinter.CTkComboBox, {"values": ["0", "1", "2"]}),
            ("Seed:", "seed", customtkinter.CTkEntry, {}),
            # ---
            ("Batch size:", "batch_size", customtkinter.CTkEntry, {}),
            ("Devices:", "devices", customtkinter.CTkEntry, {}),
            ("Max epochs:", "max_epochs", customtkinter.CTkEntry, {}),
            ("Val epoch:", "check_val_every_n_epoch", customtkinter.CTkEntry, {}),
            ("ROI length:", "roi_len", customtkinter.CTkEntry, {}),
            ("Tilt angle:", "tilt_angle", customtkinter.CTkEntry, {}),
            ("Infer overlap:", "infer_overlap", customtkinter.CTkEntry, {}),
            ("Infer batch size:", "sw_batch_size", customtkinter.CTkEntry, {}),
            ("Dimension:", "spatial_dims", customtkinter.CTkEntry, {}),
            ("Dataset length:", "dataset_len", customtkinter.CTkEntry, {}),
            ("Num workers:", "num_workers", customtkinter.CTkEntry, {}),
            ("Port id:", "port_id", customtkinter.CTkEntry, {}),
        ]

        for row, (label_text, var_name, widget_class, widget_kwargs) in enumerate(infos):
            self.get_label(
                s1_2,
                text=label_text,
                row=row,
                column=0,
                pady=(0, pady) if row == 0 else pady,
            )
            self.train_var[var_name] = tk.StringVar()
            if widget_class in [
                customtkinter.CTkOptionMenu,
                customtkinter.CTkComboBox,
                customtkinter.CTkCheckBox,
            ]:
                widget = widget_class(s1_2, variable=self.train_var[var_name], **widget_kwargs)
            else:
                widget = widget_class(s1_2, textvariable=self.train_var[var_name], **widget_kwargs)
            widget.grid(
                row=row,
                column=1,
                padx=padx,
                pady=(0, pady) if row == 0 else pady,
                sticky="ew",
            )

        # ----- frame_2, scroll_2 -----

        scroll_2 = customtkinter.CTkScrollableFrame(frame_2, label_text=" ")
        scroll_2.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="nsew")
        scroll_2.grid_rowconfigure(0, weight=1)
        scroll_2.grid_columnconfigure(0, weight=1)

        self.ablation_cfg = tk.StringVar()
        self.ablation_cfg.set(Converter.to_yaml(self.cfg_model["ablation"]))  # type: ignore[index]
        self.ablation_textbox = customtkinter.CTkTextbox(
            scroll_2, wrap="word", height=800, fg_color="transparent"
        )
        self.ablation_textbox.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
        self.ablation_textbox.insert("end", self.ablation_cfg.get())

        # ----- frame_3 -----
        frame_3 = customtkinter.CTkFrame(master, fg_color="transparent")
        frame_3.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")
        frame_3.grid_columnconfigure(0, weight=1)
        frame_3.grid_rowconfigure((0, 1), weight=1)

        # create tabview
        self.model_tabview = customtkinter.CTkTabview(
            frame_3, anchor="n", height=40
        )  # anchor="ew", fg_color="gray50"
        self.model_tabview.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
        for model in [
            "ours",
            "emnet",
            "umamba",
            "segresnet",
            "unet",
            "unetr",
            "swinunetr",
        ]:
            self.get_tabtext(model)

        # ----- textbox  -----

        self.train_textbox = customtkinter.CTkTextbox(frame_3, wrap="word", height=120)
        self.train_textbox.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        logger_name = "train"
        self.loggers[logger_name] = setup_logging(self.train_textbox, logger_name)

    def build_third_panel(self) -> None:
        """Constructs the third panel."""
        self.infer_var = {}

        self.third_panel = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        master = self.third_panel
        master.grid_columnconfigure((0), weight=1)
        master.grid_rowconfigure((1), weight=1)

        # ----- control_frame -----

        control_frame = customtkinter.CTkFrame(master)
        control_frame.grid(row=0, column=1, rowspan=2, padx=0, pady=0, sticky="nsew")
        control_frame.grid_columnconfigure(0, weight=1)

        fc_1 = customtkinter.CTkFrame(control_frame, fg_color="transparent")
        fc_1.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        fc_1.grid_columnconfigure((0, 1, 2), weight=1)
        fc_2 = customtkinter.CTkFrame(control_frame, fg_color="transparent")
        fc_2.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        fc_3 = customtkinter.CTkFrame(control_frame, fg_color="transparent")
        fc_3.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="nsew")

        buttons_info = [
            (fc_1, "Clear", self.clear_infer_event, 12, "normal", 20, 0, 0),
            (fc_1, "Stop", self.stop_cmd_event, 12, "normal", 20, 2, 0),
            (fc_2, "Run", self.run_infer_event, 14, "bold", 140, 0, 0),
        ]

        # Buttons
        for frame, text, command, size, weight, width, column, padx in buttons_info:
            button = self.create_control_button(frame, text, command, size, weight, width)
            button.grid(row=0, column=column, padx=padx, pady=0, sticky="nsew")

        self.infer_var["memo"] = tk.StringVar()
        infer_memo_entry = customtkinter.CTkEntry(fc_3, textvariable=self.infer_var["memo"])
        infer_memo_entry.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        # ----- frame_2, scroll_1 -----

        padx = (10, 20)  # type: ignore[assignment]
        pady = 5

        frame_2 = customtkinter.CTkFrame(master, fg_color="transparent")
        frame_2.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="nsew")
        frame_2.grid_columnconfigure(0, weight=1)
        frame_2.grid_rowconfigure(0, weight=1)

        scroll_1 = customtkinter.CTkScrollableFrame(frame_2, label_text=" ")
        scroll_1.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
        scroll_1.grid_columnconfigure(1, weight=1)

        infos = [
            (
                "Base dir:",
                "base_dir",
                customtkinter.CTkOptionMenu,
                {
                    "command": self.select_infer_base_event,
                    "values": ["Click to select..."],
                },
            ),
            (
                "Infer setup:",
                "infer_setup",
                customtkinter.CTkComboBox,
                {"command": self.select_infer_event, "values": [""]},
            ),
            ("Devices:", "devices", customtkinter.CTkEntry, {}),
            ("Infer overlap:", "infer_overlap", customtkinter.CTkEntry, {}),
            ("Infer batch size:", "sw_batch_size", customtkinter.CTkEntry, {}),
        ]

        for row, (label_text, var_name, widget_class, widget_kwargs) in enumerate(infos):
            self.get_label(
                scroll_1,
                text=label_text,
                row=row,
                column=0,
                pady=(0, pady) if row == 0 else pady,
            )
            self.infer_var[var_name] = tk.StringVar()
            if widget_class in [
                customtkinter.CTkOptionMenu,
                customtkinter.CTkComboBox,
                customtkinter.CTkCheckBox,
            ]:
                widget = widget_class(scroll_1, variable=self.infer_var[var_name], **widget_kwargs)
            else:
                widget = widget_class(
                    scroll_1, textvariable=self.infer_var[var_name], **widget_kwargs
                )
            if var_name == "infer_setup":
                self.infer_setup_entry = widget
            widget.grid(
                row=row,
                column=1,
                padx=padx,
                pady=(0, pady) if row == 0 else pady,
                sticky="ew",
            )

        # ----- textbox -----

        self.infer_textbox = customtkinter.CTkTextbox(master, width=200, wrap="word")
        self.infer_textbox.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        logger_name = "infer"
        self.loggers[logger_name] = setup_logging(self.infer_textbox, logger_name)

    def build_fourth_panel(self) -> None:
        """Constructs the fourth panel."""
        self.fourth_panel = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        master = self.fourth_panel
        master.grid_columnconfigure((0), weight=1)
        master.grid_rowconfigure((1), weight=1)
        frame_1 = customtkinter.CTkFrame(master)
        frame_1.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="nsew")
        frame_1.grid_columnconfigure((0), weight=1)
        subframe_1 = customtkinter.CTkFrame(frame_1)
        subframe_1.grid(row=0, column=0, padx=(20, 10), pady=10, sticky="nsew")
        subframe_1.grid_columnconfigure(1, weight=1)
        subframe_2 = customtkinter.CTkFrame(frame_1, width=100)
        subframe_2.grid(row=0, column=1, padx=(10, 20), pady=10, sticky="nsew")

        padx = (10, 20)  # type: ignore[assignment]
        pady = 2

        # subframe_1
        xyz_width = 70
        xyz_max = 100
        initial_img = to_ctkimg(np.ones((200, 200), dtype=np.uint8) * 255)
        infos = [
            ("X", self.entry_x_event, self.update_x_event),
            ("Y", self.entry_y_event, self.update_y_event),
            ("Z", self.entry_z_event, self.update_z_event),
        ]
        for row, (axis, entry_command, update_command) in enumerate(infos):
            name = axis.lower()
            self.get_label(
                subframe_1,
                text=f"{axis}:",
                row=row,
                column=0,
                pady=(10, pady) if row == 0 else pady,
            )

            var = tk.StringVar(value="0")
            var.trace_add("write", entry_command)
            setattr(self, f"{name}_var", var)

            slider = customtkinter.CTkSlider(
                subframe_1,
                from_=0,
                to=xyz_max,
                number_of_steps=xyz_max,
                command=update_command,
            )
            slider.grid(
                row=row,
                column=1,
                padx=(10, 0),
                pady=(10, pady) if row == 0 else pady,
                sticky="ew",
            )
            setattr(self, f"{name}_slider", slider)

            entry = customtkinter.CTkComboBox(
                subframe_1,
                width=xyz_width,
                variable=var,
                values=["Center", "StartR", "StopR"],
                command=entry_command,
            )
            entry.grid(row=row, column=2, padx=padx, pady=(10, pady) if row == 0 else pady)

        # subframe_2
        infos = [
            ("Enhance", (10, pady)),  # type: ignore[list-item]
            ("Invert", (pady, pady)),  # type: ignore[list-item]
            ("FFT", (pady, 10)),  # type: ignore[list-item]
        ]
        radio_var = tk.IntVar(value=0)
        for row, (text, _pady) in enumerate(infos):  # type: ignore[misc]
            value = row + 1
            name = text.lower()
            self.get_label(subframe_2, text="", row=row, column=0, pady=_pady)
            radio_button = customtkinter.CTkRadioButton(
                subframe_2,
                text=text,
                variable=radio_var,
                value=value,
                command=getattr(self, f"{name}_event"),
            )
            radio_button.grid(row=row, column=1, padx=10, pady=_pady, sticky="ew")

        # create tabview
        tabview = customtkinter.CTkTabview(master)
        tabview.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # ----- subframe -----

        # self.noisy_show, self.noisy_raw, self.noisy_enhance, self.noisy_invert, self.noisy_fft
        for i in ["noisy", "denoised", "clean"]:
            for j in ["show", "raw", "enhance", "invert", "fft"]:
                setattr(self, f"{i}_{j}", None)

        tab_infos = [
            ("Noisy", "noisy"),
            ("Denoised", "denoised"),
            ("Clean / Label", "clean"),
        ]

        xyz_infos = [
            ("XY", (0, 0), (20, 10), (20, 10)),
            ("YZ", (0, 1), (10, 20), (20, 10)),
            ("XZ", (1, 0), (20, 10), (10, 20)),
        ]

        for tab_text, tab_name in tab_infos:
            tabview.add(tab_text)
            tab = tabview.tab(tab_text)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(1, weight=1)

            button = customtkinter.CTkButton(
                tab,
                text="Load",
                font=customtkinter.CTkFont(size=14, weight="bold"),
                command=getattr(self, f"load_{tab_name}_event"),  # self.load_noisy_event
            )
            button.grid(row=0, column=0, padx=20, pady=5, sticky="nsew")

            scroll_frame = customtkinter.CTkScrollableFrame(tab, label_text=None)
            scroll_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
            scroll_frame.grid_columnconfigure((0, 1), weight=1)
            scroll_frame.grid_rowconfigure((0, 1), weight=1)

            for label, (row, col), _padx, _pady in xyz_infos:
                frame = customtkinter.CTkLabel(
                    scroll_frame,
                    image=initial_img,
                    text=label,
                    font=customtkinter.CTkFont(size=14, weight="bold"),
                    anchor="n",
                    compound="bottom",
                )
                frame.grid(row=row, column=col, padx=_padx, pady=_pady, sticky="nsew")
                setattr(self, f"frame_{label.lower()}_{tab_name}", frame)  # self.frame_xy_noisy

    # ===== event-basic =====

    def stop_cmd_event(self) -> None:
        """Stops a running process."""
        if self.process and self.process.poll() is None:  # Check if the process is running
            self.stop_event.set()  # Signal the thread to stop

    def on_close(self) -> None:
        """Handles the application's close event."""
        if self.process and self.process.poll() is None:
            if messagebox.askyesno("Exit Confirmation", "The program is running. Confirm exit?"):
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)  # type: ignore[attr-defined]
                while True:
                    if self.process.poll() is not None:
                        self.destroy()
                        break
            else:
                pass
        else:
            self.destroy()

    # ===== event-navigation =====

    def select_panel_event(self, name: str) -> None:
        """Selects a panel by name and updates the UI accordingly."""
        maps = {
            "first_panel": (self.first_panel_button, self.first_panel),
            "second_panel": (self.second_panel_button, self.second_panel),
            "third_panel": (self.third_panel_button, self.third_panel),
            "fourth_panel": (self.fourth_panel_button, self.fourth_panel),
        }

        # Set button color for selected button and show the corresponding panel
        for panel_name, (button, panel) in maps.items():
            if name == panel_name:
                button.configure(fg_color=("gray75", "gray25"))
                panel.grid(row=0, column=1, sticky="nsew")
            else:
                button.configure(fg_color="transparent")
                panel.grid_forget()

    def change_appearance_mode_event(self, inp: str) -> None:
        """Changes the appearance mode of the application."""
        customtkinter.set_appearance_mode(inp)

    def change_scaling_event(self, inp: str) -> None:
        """Adjusts the widget scaling based on the input percentage string."""
        inp = int(inp.rstrip("%")) / 100  # type: ignore[assignment]
        customtkinter.set_widget_scaling(inp)

    # ===== event-first =====

    def load_pre_config_event(self) -> None:
        """Loads a configuration file and displays its contents."""
        file_path = filedialog.askopenfilename(
            title="Select Configuration File for Preprocessing",
            filetypes=[("Config files", "*.yaml"), ("All files", "*.*")],
        )

        if file_path:
            self.pre_config = OmegaConf.load(file_path)
            self.loggers["pre"].info(f"Load {file_path}:\n\n" + Converter.to_yaml(self.pre_config))
            for key, var in self.pre_var.items():
                var.set(self.pre_config.get(key, ""))  # type: ignore[arg-type]

    def clear_pre_event(self) -> None:
        """Clears the pre_textbox and resets pre_var values."""
        self.pre_textbox.delete("1.0", "end")
        for key, var in self.pre_var.items():
            var.set(0 if key == "apply_deconv" else "")  # type: ignore[arg-type]

    def _save_pre_event(self, yaml_path: Union[str, Path]) -> None:
        """Saves configuration to a YAML file."""
        yaml_data: Dict = {}
        for key, var in self.pre_var.items():
            value = var.get()
            if value in [None, "None", ""]:
                yaml_data[key] = None
            elif key == "apply_deconv":
                yaml_data[key] = value == 1
            elif key in {"mode", "seed", "roi_len", "defocus"}:
                yaml_data[key] = int(value)
            elif key in {"voxel_size", "overlap"}:
                yaml_data[key] = float(value)
            else:
                yaml_data[key] = value

        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(yaml_data), yaml_path)
        self.loggers["pre"].info(f"Save {yaml_path}\n\n")

    def save_pre_event(self) -> None:
        """Prompts user to select a file path and saves configuration."""
        file_path = filedialog.asksaveasfilename(
            title="Select Directory for Saving",
            initialfile="preprocess_custom",
            defaultextension=".yaml",
            filetypes=[("YAML Files", "*.yaml"), ("All Files", "*.*")],
        )
        self._save_pre_event(file_path)

    def run_pre_event(self) -> None:
        """Executes the shell command."""
        self.run_pre_path = step_1
        self.pre_config_path = (
            Path(self.pre_var["base_dir"].get()) / "0_input" / "config" / "preprocess_running.yaml"
        )
        self._save_pre_event(self.pre_config_path)

        cmd = f"{run_python} {self.run_pre_path} --config_file {self.pre_config_path}"
        self.loggers["pre"].info(f"Run {cmd}\n\n")
        self.run_cmd(self.pre_textbox, cmd)

    def select_pre_base_event(self, *args: str) -> None:
        """Opens a directory selection dialog and updates the base_dir variable."""
        if directory := filedialog.askdirectory():
            self.pre_var["base_dir"].set(directory)

    def val_set_event(self) -> None:
        """Configures validation settings based on the number of input data files."""
        input_dir = Path(self.pre_var["base_dir"].get()) / "0_input" / "data" / "noisy" / "all"
        input_paths = sorted(input_dir.glob("*.mrc"))

        if len(input_paths) == 1:
            text = "Detected that the number of input data is 1, \n\
                please enter the validation ratio."
            height = 200
        else:
            text = f"Detected that the number of input data is {len(input_paths)}, \n\
                please enter either the validation ratio or the indices for validation.\n\n"
            for i in range(len(input_paths)):
                text += f"{i}:{input_paths[i].name}\n"
            height = 600

        dialog = CustomInputDialog(
            text=text, title="Validation Configuration", width=350, height=height
        )
        self.pre_var["val_setup"].set(dialog.get_input())

    # ===== event-second =====
    def load_train_config_event(self) -> None:
        """Loads a configuration file and displays its contents."""
        file_path = filedialog.askopenfilename(
            title="Select Configuration File for Training",
            filetypes=[("Config files", "*.yaml"), ("All files", "*.*")],
        )
        if file_path:
            self.train_config = OmegaConf.load(file_path)
            self.loggers["train"].info(
                f"Load {file_path}:\n\n" + Converter.to_yaml(self.train_config)
            )
            if self.train_config.basic.get("pretrained_path"):
                self.pretrain_check.select()

            for key, var in self.train_var.items():
                var.set(self.train_config.basic.get(key, ""))

            model_name = self.train_var["model"].get() or None
            if model_name in self.train_config:
                self.select_model_event(model_name)  # type: ignore[arg-type]
                self.train_tabtext[model_name].delete("1.0", "end")
                self.train_tabtext[model_name].insert(
                    "end",
                    Converter.to_yaml(self.train_config[model_name]),  # type: ignore[index]
                )

            if "ablation" in self.train_config:
                self.ablation_textbox.delete("1.0", "end")
                self.ablation_textbox.insert(
                    "end",
                    Converter.to_yaml(self.train_config["ablation"]),
                )

    def clear_train_event(self) -> None:
        """Clears the train_textbox and resets train_var values."""
        self.train_textbox.delete("1.0", "end")
        for var in self.train_var.values():
            var.set("")

        for model_name, tabtext in self.train_tabtext.items():
            tabtext.delete("1.0", "end")
            tabtext.insert("end", self.train_cfg[model_name].get())
            tabtext.configure(font=customtkinter.CTkFont(size=14))

        self.ablation_textbox.delete("1.0", "end")
        self.ablation_textbox.insert("end", self.ablation_cfg.get())

    def _save_train_event(self, yaml_path: Union[str, Path]) -> None:
        """Saves configuration to a YAML file."""
        model_name = self.train_var["model"].get() or "ours"
        model_cfg = OmegaConf.create(self.train_tabtext[model_name].get("1.0", "end-1c"))
        ablation_cfg = OmegaConf.create(self.ablation_textbox.get("1.0", "end-1c"))

        yaml_data = {"basic": {}, model_name: model_cfg, "ablation": ablation_cfg}
        for key, var in self.train_var.items():
            value = var.get()
            if value in [None, "None", ""]:
                value = None
            elif key == "devices":
                value = Converter.to_int_list(value)
            elif key in {
                "mode",
                "seed",
                "batch_size",
                "max_epochs",
                "check_val_every_n_epoch",
                "roi_len",
                "tilt_angle",
                "sw_batch_size",
                "spatial_dims",
                "dataset_len",
                "num_workers",
                "port_id",
            }:
                value = int(value)  # type: ignore[assignment]
            elif key == "infer_overlap":
                value = float(value)  # type: ignore[assignment]
            yaml_data["basic"][key] = value

        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(yaml_data), yaml_path)
        self.loggers["train"].info(f"Save {yaml_path}\n\n")

    def save_train_event(self) -> None:
        """Prompts user to select a file path and saves configuration."""
        file_path = filedialog.asksaveasfilename(
            title="Select Directory for Saving",
            initialfile="train_custom",
            defaultextension=".yaml",
            filetypes=[("YAML Files", "*.yaml"), ("All Files", "*.*")],
        )

        self._save_train_event(file_path)

    def run_train_event(self) -> None:
        """Executes the shell command."""
        self.run_train_path = step_2
        self.train_config_path = (
            Path(self.train_var["base_dir"].get()) / "0_input" / "config" / "train_running.yaml"
        )
        self._save_train_event(self.train_config_path)

        base_cmd = f"{run_python} {self.run_train_path} --config_file {self.train_config_path}"

        if self.train_var["seed"].get() == "40":  # TODO
            cmd = ""
            for s in [40, 41, 42]:
                cmd += f"{base_cmd} --seed {s} && "
            cmd += f"{run_python} {step_3} --config_file {self.train_config_path}"
        else:
            cmd = base_cmd

        port_id = self.train_var["port_id"].get()
        if port_id not in [None, "None", ""]:
            logdir = Path(self.train_var["base_dir"].get()).resolve() / "2_train"
            cmd = f"tensorboard --logdir={logdir} --bind_all --port={port_id} & " + cmd

        if "cmd" in self.train_var["memo"].get():
            self.loggers["train"].info(cmd)
        else:
            self.loggers["train"].info(f"Run {cmd}\n\n")
            self.run_cmd(self.train_textbox, cmd)

    def pretrain_event(self) -> None:
        """Opens a file dialog to select pretrained weights."""
        file_path = filedialog.askopenfilename(
            title="Select Pretrained Weights for Training",
            filetypes=[("Weights files", "*.ckpt"), ("All files", "*.*")],
        )
        if file_path:
            self.train_var["pretrained_path"].set(file_path)

    def select_model_event(self, value: str) -> None:
        """Updates UI to reflect the selected model."""
        for model_name in self.train_tabtext.keys():
            if value == model_name:
                self.model_tabview.set(value)
                self.train_var["model"].set(value)
                self.train_tabtext[model_name].configure(
                    font=customtkinter.CTkFont(size=14, weight="bold")
                )
            else:
                self.train_tabtext[model_name].configure(font=customtkinter.CTkFont(size=14))

    def select_train_base_event(self, *args: str) -> None:
        """Opens a directory selection dialog and updates the base_dir variable."""
        if directory := filedialog.askdirectory():
            self.train_var["base_dir"].set(directory)

    def select_train_json_event(self, *args: str) -> None:
        """Opens a file dialog to select a JSON file."""
        file_path = filedialog.askopenfilename(
            title="Select Dataset File for Training",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if file_path:
            self.train_var["json_path"].set(file_path)

    # ===== event-third =====
    def select_infer_base_event(self, *args: str) -> None:
        """Opens a directory selection dialog and updates the base_dir variable."""
        if directory := filedialog.askdirectory():
            self.infer_var["base_dir"].set(directory)

            ckpt_paths = sorted(Path(directory).rglob("*.ckpt"))
            ckpt_paths = [str(p) for p in ckpt_paths]  # type: ignore[misc]
            dirs = list({p.split("checkpoints")[0] for p in ckpt_paths})  # type: ignore[attr-defined]

            self.infer_setup_entry.configure(values=dirs + ckpt_paths)

    def clear_infer_event(self) -> None:
        """Clears the infer_textbox and resets infer_var values."""
        self.infer_textbox.delete("1.0", "end")
        for var in self.infer_var.values():
            var.set("")

    def select_infer_event(self, *args: str) -> None:
        """Loads configuration based on the selected inference setup."""
        infer_setup = self.infer_var["infer_setup"].get()
        file_path = Path(infer_setup.split("checkpoints")[0]) / "trained.yaml"
        self.trained_path = (
            file_path
            if file_path.exists()
            else Path(infer_setup.split("version_")[0]) / "training.yaml"
        )

        trained_cfg = OmegaConf.load(self.trained_path)
        self.loggers["infer"].info(f"Load {self.trained_path}\n\n")

        # Update inference variables with loaded configuration
        for var in ["devices", "infer_overlap", "sw_batch_size"]:
            self.infer_var[var].set(getattr(trained_cfg.basic, var))

    def run_infer_event(self) -> None:
        """Executes the shell command."""
        self.run_infer_path = step_2
        cmd = f"{run_python} {self.run_infer_path} --config_file {self.trained_path} "

        # --infer
        infer_setup = self.infer_var["infer_setup"].get()
        if ".ckpt" not in infer_setup:
            infer_setup = int(infer_setup.split("version_")[-1].split("/")[0])  # type: ignore[assignment]
        cmd += f"--infer {infer_setup} "

        # --devices
        devices = Converter.to_int_list(self.infer_var["devices"].get())
        cmd = cmd + "--devices "
        if isinstance(devices, int):
            cmd += f"{devices} "
        else:
            cmd += " ".join(map(str, devices)) + " "

        # --infer_overlap
        infer_overlap = self.infer_var["infer_overlap"].get()
        if infer_overlap not in [None, "None", ""]:
            cmd += f"--infer_overlap {infer_overlap} "

        # --sw_batch_size
        sw_batch_size = self.infer_var["sw_batch_size"].get()
        if sw_batch_size not in [None, "None", ""]:
            cmd += f"--sw_batch_size {sw_batch_size} "

        if "cmd" in self.infer_var["memo"].get():
            self.loggers["infer"].info(cmd)
        else:
            self.loggers["infer"].info(f"Run {cmd}\n\n")
            self.run_cmd(self.infer_textbox, cmd)

    # ===== event-fourth =====
    def load_img_event(self, event_type: str) -> None:
        """Loads and displays a tomographic file based on the event type."""
        file_path = filedialog.askopenfilename(
            title="Select Tomo File",
            filetypes=[
                ("Tomo files", "*.mrc"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            arr = get_scaled(read_mrc(file_path))
            arr_int = to_int(arr)

            setattr(self, f"{event_type}_raw", arr)  # self.noisy_raw = arr
            setattr(self, f"{event_type}_show", arr_int)  # self.noisy_show = arr_int

            self.z_max, self.y_max, self.x_max = (dim - 1 for dim in arr.shape)
            self.z_mid, self.y_mid, self.x_mid = (dim // 2 for dim in arr.shape)

            infos = [
                ("x", self.update_x_event, self.x_max, self.x_mid, "yz"),
                ("y", self.update_y_event, self.y_max, self.y_mid, "xz"),
                ("z", self.update_z_event, self.z_max, self.z_mid, "xy"),
            ]

            for axis, update_func, max_val, mid_val, plane in infos:
                update_func(mid_val)

                slider = getattr(self, f"{axis}_slider")
                slider.configure(to=max_val, number_of_steps=max_val)
                slider.set(mid_val)

                frame = getattr(self, f"frame_{plane}_{event_type}")
                frame.configure(image=to_ctkimg(globals()[f"get_{plane}"](arr_int, mid_val)))

    def load_noisy_event(self) -> None:
        """Loads and displays noisy image."""
        self.load_img_event("noisy")

    def load_denoised_event(self) -> None:
        """Loads and displays denoised image."""
        self.load_img_event("denoised")

    def load_clean_event(self) -> None:
        """Loads and displays clean image."""
        self.load_img_event("clean")

    def update_axis_event(self, axis: str, value: int) -> None:
        """Updates the display based on the specified axis."""
        maps = {"x": ("yz", get_yz), "y": ("xz", get_xz), "z": ("xy", get_xy)}

        value = int(value)
        getattr(self, f"{axis}_var").set(f"{value}")

        if axis in maps:
            plane, get_func = maps[axis]
            infos = [
                ("noisy_show", f"frame_{plane}_noisy"),
                ("denoised_show", f"frame_{plane}_denoised"),
                ("clean_show", f"frame_{plane}_clean"),
            ]

            for show_attr, frame_attr in infos:
                show = getattr(self, show_attr)
                if show is not None:
                    frame = getattr(self, frame_attr)
                    frame.configure(image=to_ctkimg(get_func(show, value)))

    def update_x_event(self, value: int) -> None:
        """Updates the display for the x-axis."""
        self.update_axis_event("x", value)

    def update_y_event(self, value: int) -> None:
        """Updates the display for the y-axis."""
        self.update_axis_event("y", value)

    def update_z_event(self, value: int) -> None:
        """Updates the display for the z-axis."""
        self.update_axis_event("z", value)

    def auto_slide_event(self, axis: str) -> None:
        """Automatically slides the specified axis slider."""
        running_attr = f"{axis}_running"
        run_value_attr = f"{axis}_run_value"
        direction_attr = f"{axis}_direction"
        max_attr = f"{axis}_max"
        slider_attr = f"{axis}_slider"
        update_event = getattr(self, f"update_{axis}_event")

        if getattr(self, running_attr):
            run_value = getattr(self, run_value_attr) + getattr(self, direction_attr)
            max_val = getattr(self, max_attr)
            run_value = np.clip(run_value, 0, max_val, dtype=np.int16)
            setattr(self, run_value_attr, run_value)

            getattr(self, slider_attr).set(run_value)
            update_event(run_value)

            if run_value >= max_val:
                setattr(self, direction_attr, -10)
            elif run_value <= 0:
                setattr(self, direction_attr, 10)

            self.after(100, lambda: self.auto_slide_event(axis))

    def auto_slide_x_event(self) -> None:
        """Triggers the automatic sliding event for the x-axis."""
        self.auto_slide_event("x")

    def auto_slide_y_event(self) -> None:
        """Triggers the automatic sliding event for the y-axis."""
        self.auto_slide_event("y")

    def auto_slide_z_event(self) -> None:
        """Triggers the automatic sliding event for the z-axis."""
        self.auto_slide_event("z")

    def entry_axis_event(self, axis: str, click: str, *args: str) -> None:
        """Handles entry events for the specified axis."""
        var = getattr(self, f"{axis}_var")
        mid = getattr(self, f"{axis}_mid")
        max_val = getattr(self, f"{axis}_max")
        slider = getattr(self, f"{axis}_slider")
        update_event = getattr(self, f"update_{axis}_event")
        running_attr = f"{axis}_running"
        run_value_attr = f"{axis}_run_value"
        direction_attr = f"{axis}_direction"
        auto_slide_event = getattr(self, f"auto_slide_{axis}_event")

        if "Center" in [click, var.get()]:
            update_event(mid)
        elif "StartR" in [click, var.get()]:
            setattr(self, running_attr, True)
            setattr(self, run_value_attr, 0)
            setattr(self, direction_attr, 10)
            auto_slide_event()
        elif "StopR" in [click, var.get()]:
            setattr(self, running_attr, False)
        else:
            if var.get():
                value = int(var.get())
                if 0 <= value <= max_val:
                    slider.set(value)
                    update_event(value)

    def entry_x_event(self, click: str, *args: str) -> None:
        """Handles entry events for the x-axis."""
        self.entry_axis_event("x", click, *args)

    def entry_y_event(self, click: str, *args: str) -> None:
        """Handles entry events for the y-axis."""
        self.entry_axis_event("y", click, *args)

    def entry_z_event(self, click: str, *args: str) -> None:
        """Handles entry events for the z-axis."""
        self.entry_axis_event("z", click, *args)

    def process_event(self, process: str) -> None:
        """Process images by applying a specified transformation function."""
        process_func = {
            "enhance": get_intensity_rescaled,
            "invert": get_inverted,
            "fft": get_fft,
        }.get(process)

        for img in ["noisy", "denoised", "clean"]:
            img_raw = getattr(self, f"{img}_raw")
            img_processed = getattr(self, f"{img}_{process}")

            if img_processed is not None:
                setattr(self, f"{img}_show", img_processed)
            elif img_raw is not None:
                img_processed = to_int(process_func(img_raw))  # type: ignore[operator,misc]
                setattr(self, f"{img}_show", img_processed)
                setattr(self, f"{img}_{process}", img_processed)

        self.update_x_event(self.x_mid)
        self.update_y_event(self.y_mid)
        self.update_z_event(self.z_mid)

    def enhance_event(self) -> None:
        """Triggers image processing events."""
        self.process_event("enhance")

    def invert_event(self) -> None:
        """Triggers image processing events."""
        self.process_event("invert")

    def fft_event(self) -> None:
        """Triggers image processing events."""
        self.process_event("fft")


def launch_gui() -> None:
    """Initializes and runs the GUI application."""
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
