import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from dacite import from_dict, Config
from typing import Any, List



def ensure_directory_exists(file_path: Path) -> None:
    """
    Ensures that the directory for the given file_path exists.
    If it doesn't, the directory (and any necessary parent directories) is created.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)


class Console:
    """
    A simple console utility for error output.
    This can be extended or replaced with a full logging mechanism if needed.
    """

    @staticmethod
    def error(message: str, error: Exception) -> None:
        raise error(message)


console = Console()


def deep_merge(default: dict[Any, Any], override: dict[Any, Any]) -> dict[Any, Any]:
    """
    deep merge two dictionaries.：
    for the same key，if mapping to a dictionary，merge；
    otherwise override
    """
    result = default.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result




@dataclass
class JsonConfig:
    """
    A Python dataclass bundled with methods to save and load itself from a .json file.

    This class can be used as a base class for configuration classes.
    It provides methods to:
      - Convert the instance into a dictionary and write it to a JSON file.
      - Instantiate a new instance from data stored in a JSON file.

    Subclasses should define their own fields.
    """

    def to_json(self, file_path: Path) -> None:
        """
        Converts the dataclass instance to a dictionary and saves it as a .json file.

        Args:
            file_path: The path to the JSON file to write.
                       The file path must end with ".json". If the file does not exist, the directory
                       will be created; if it exists, it will be overwritten.

        Raises:
            ValueError: If the file path does not have a '.json' extension.
        """


        # Ensure the file extension is .json
        if file_path.suffix.lower() != ".json":
            message = (
                f"Invalid file path provided. Expected a path ending with '.json', but got {file_path}."
            )
            console.error(message=message, error=ValueError)

        # Ensure that the output directory exists
        ensure_directory_exists(file_path)

        instance_data = asdict(self)


        # Write the instance data to the JSON file with indentation for readability
        with open(file_path, "w") as json_file:
            json.dump(instance_data, json_file)

    @classmethod
    def from_json(cls, file_path: Path) -> "JsonConfig":
        """
        Instantiates the class using data loaded from the provided .json file.

        This method reads the JSON file, converts it to a dictionary, and then uses dacite
        to instantiate the class. It disables dacite's built-in type checking, so you may need to
        add extra validation if required.

        Args:
            file_path: The path to the JSON file to read.

        Returns:
            A new dataclass instance created using the data read from the JSON file.

        Raises:
            ValueError: If the file path does not have a '.json' extension.
        """
        # Ensure the file extension is .json
        if file_path.suffix.lower() != ".json":
            message = (
                f"Invalid file path provided. Expected a path ending with '.json', but got {file_path}."
            )
            console.error(message=message, error=ValueError)

        # Disable dacite's built-in type checking
        class_config = Config(check_types=False)

        # Open and read the JSON file
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        # Convert the loaded data to a dictionary
        config_dict: dict[Any, Any] = dict(data)

        # Use dacite to instantiate the dataclass from the dictionary
        instance = from_dict(data_class=cls, data=config_dict, config=class_config)
        return instance


@dataclass
class MyConfig(JsonConfig):
    """
       Configuration class with default values.
    """
    # Main settings
    nplanes: int = 1
    nchannels: int = 1
    functional_chan: int = 1
    tau: float = 1.0
    force_sktiff: bool = False
    fs: float = 10.0
    do_bidiphase: bool = False
    bidiphase: int = 0
    bidi_corrected: bool = False
    frames_include: int = -1
    multiplane_parallel: bool = False
    ignore_flyback: List[int] = field(default_factory=list)

    # File input/output settings
    fast_disk: List[str] = field(default_factory=list)
    delete_bin: bool = False
    mesoscan: bool = False
    bruker: bool = False
    bruker_bidirectional: bool = False
    h5py: List[str] = field(default_factory=list)
    h5py_key: str = "data"
    nwb_file: str = ""
    nwb_driver: str = ""
    nwb_series: str = ""
    save_path0: List[str] = field(default_factory=list)
    save_folder: List[str] = field(default_factory=list)
    look_one_level_down: bool = False
    subfolders: List[str] = field(default_factory=list)
    move_bin: bool = False

    # Output settings
    preclassify: float = 0.0
    save_nwb: bool = False
    save_mat: bool = False
    combined: bool = True
    aspect: float = 1.0
    report_time: bool = True

    # Registration settings
    do_registration: bool = True
    align_by_chan: int = 1
    nimg_init: int = 300
    batch_size: int = 500
    maxregshift: float = 0.1
    smooth_sigma: float = 1.15
    smooth_sigma_time: float = 0.0
    keep_movie_raw: bool = False
    two_step_registration: bool = False
    reg_tif: bool = False
    reg_tif_chan2: bool = False
    subpixel: int = 10
    th_badframes: float = 1.0
    norm_frames: bool = True
    force_refImg: bool = False
    pad_fft: bool = False

    # 1P registration settings
    one_p_reg: bool = False
    spatial_hp_reg: int = 42
    pre_smooth: float = 0.0
    spatial_taper: float = 40.0

    # Non-rigid registration settings
    nonrigid: bool = True
    block_size: List[int] = field(default_factory=lambda: [128, 128])
    snr_thresh: float = 1.2
    maxregshiftNR: float = 5.0

    # ROI detection settings
    roidetect: bool = True
    sparse_mode: bool = True
    spatial_scale: int = 0
    connected: bool = True
    threshold_scaling: float = 1.0
    spatial_hp_detect: int = 25
    max_overlap: float = 0.75
    high_pass: int = 100
    smooth_masks: bool = True
    max_iterations: int = 20
    nbinned: int = 5000
    denoise: bool = False

    # Cellpose Detection settings
    anatomical_only: int = 0
    diameter: int = 0
    cellprob_threshold: float = 0.0
    flow_threshold: float = 1.5
    spatial_hp_cp: int = 0
    pretrained_model: str = "cyto"

    # Signal extraction settings
    neuropil_extract: bool = True
    allow_overlap: bool = False
    min_neuropil_pixels: int = 350
    inner_neuropil_radius: int = 2
    lam_percentile: int = 50

    # Spike deconvolution settings
    spikedetect: bool = True
    neucoeff: float = 0.7
    baseline: str = "maximin"
    win_baseline: float = 60.0
    sig_baseline: float = 10.0
    prctile_baseline: int = 8

    # Classification settings
    soma_crop: bool = True
    use_builtin_classifier: bool = False
    classifier_path: str = ""

    # Channel 2 specific settings
    chan2_thres: int = 0

    # Miscellaneous settings
    suite2p_version: str = ""


def main():
    input_file = Path("ops.json")
    output_file = Path(r"C:\Users\Kaze\Desktop\test.json")

    try:
        config = MyConfig.from_json(input_file)
    except Exception as e:
        print(f"loading error：{e}")
        return

    config.nplanes = config.nplanes + 1

    try:
        config.to_json(output_file)
    except Exception as e:
        print(f"saving error：{e}")
        return

    print(f"successfully loaded from '{input_file}' and saved to '{output_file}'.")

if __name__ == "__main__":
    main()



