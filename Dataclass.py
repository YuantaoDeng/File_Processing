import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from dacite import from_dict, Config




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
        json_formatting = {
            # Main settings
            "nplanes": 1,  # Each tiff has this many planes in sequence
            "nchannels": 1,  # Each tiff has this many channels per plane
            "functional_chan": 1,  # 1-based index; e.g., 1 means first channel used for functional ROIs
            "tau": 1.0,  # Timescale of the sensor (in seconds) for deconvolution kernel
            "force_sktiff": False,  # Whether or not to use scikit-image for reading tiffs
            "fs": 10.0,  # Sampling rate (per plane)
            "do_bidiphase": False,  # Whether to compute bidirectional phase offset (for 2P recordings)
            "bidiphase": 0,  # Bidirectional phase offset; if non-zero, this offset is applied to all frames
            "bidi_corrected": False,  # Specifies whether bidi correction is performed
            "frames_include": -1,  # If > 0, only process this many frames (useful for testing)
            "multiplane_parallel": False,  # Specifies whether to run the pipeline in parallel (e.g., on a server)
            "ignore_flyback": [],  # List of plane indices to ignore as flyback planes

            # File input/output settings
            "fast_disk": [],  # List of paths for storing temporary binary files
            "delete_bin": False,  # Whether to delete binary files created during registration
            "mesoscan": False,  # Whether the file is a ScanImage mesoscope recording
            "bruker": False,  # Whether provided tiff files are single-page BRUKER tiffs
            "bruker_bidirectional": False,  # Whether BRUKER files are bidirectional multiplane recordings
            "h5py": [],  # List of paths to h5py files (overrides .ops[data_path] if provided)
            "h5py_key": "data",  # Key to access the data array in the h5py file
            "nwb_file": "",  # Path to the NWB file used as input
            "nwb_driver": "",  # Path to the NWB driver (if necessary)
            "nwb_series": "",  # Name of the TwoPhotonSeries values to retrieve from the NWB file
            "save_path0": [],  # List of paths where pipeline results are saved; defaults to first element if empty
            "save_folder": [],  # List of directory names under which results are saved (default: "suite2p")
            "look_one_level_down": False,  # Whether to search in all subfolders for tiff files
            "subfolders": [],  # Specifies subfolders to search within (used when look_one_level_down is True)
            "move_bin": False,  # If True, move the created binary file to the location specified in fast_disk

            # Output settings
            "preclassify": 0.0,  # Apply classifier before signal extraction with this probability threshold
            "save_nwb": False,  # Whether to save output as an NWB file
            "save_mat": False,  # Whether to save the results in MATLAB format ("Fall.mat")
            "combined": True,  # Combine results across planes into a "combined" folder
            "aspect": 1.0,  # Ratio of um/pixels in X to um/pixels in Y for proper GUI display
            "report_time": True,  # Whether to return a timing dictionary for each processing stage

            # Registration settings
            "do_registration": True,  # Whether to perform registration
            "align_by_chan": 1,  # Which channel to use for alignment (1-based indexing)
            "nimg_init": 300,  # Number of frames to compute the reference image for registration
            "batch_size": 500,  # Number of frames to register simultaneously in each batch
            "maxregshift": 0.1,  # Maximum shift as a fraction of the frame size
            "smooth_sigma": 1.15,  # Std. dev. for Gaussian smoothing during registration (in pixels)
            "smooth_sigma_time": 0,  # Temporal smoothing sigma (in frames)
            "keep_movie_raw": False,  # Whether to keep the binary file of non-registered frames
            "two_step_registration": False,  # Whether to run registration twice (for low SNR data)
            "reg_tif": False,  # Whether to write the registered binary to tiff files
            "reg_tif_chan2": False,  # Whether to write tiff files for the non-functional channel
            "subpixel": 10,  # Precision of subpixel registration (1/subpixel steps)
            "th_badframes": 1.0,  # Threshold for excluding frames (for cropping)
            "norm_frames": True,  # Whether to normalize frames when detecting shifts
            "force_refImg": False,  # Specifies whether to force the use of a stored refImg
            "pad_fft": False,  # Whether to pad the image during FFT registration

            # 1P registration settings
            "1Preg": False,  # Whether to perform high-pass spatial filtering/tapering for 1P data
            "spatial_hp_reg": 42,  # Window (in pixels) for spatial high-pass filtering before registration
            "pre_smooth": 0,  # Std. dev. for Gaussian smoothing applied before high-pass filtering
            "spatial_taper": 40,
            # Number of pixels to ignore on the edges (for FFT padding, do not set below 3*smooth_sigma)

            # Non-rigid registration settings
            "nonrigid": True,  # Whether to perform non-rigid registration
            "block_size": [128, 128],  # Size of blocks for non-rigid registration (in pixels)
            "snr_thresh": 1.2,  # Phase correlation SNR threshold for accepting a block shift
            "maxregshiftNR": 5.0,  # Maximum allowed shift (in pixels) for a block relative to the rigid shift

            # ROI detection settings
            "roidetect": True,  # Whether to run ROI detection and extraction
            "sparse_mode": True,  # Whether to use sparse_mode cell detection
            "spatial_scale": 0,  # Optimal scale of the recording in pixels (0 to auto-determine)
            "connected": True,  # Whether to require ROIs to be fully connected
            "threshold_scaling": 1.0,  # Scaling factor for the detection threshold (higher -> fewer ROIs)
            "spatial_hp_detect": 25,  # Window for spatial high-pass filtering before ROI detection
            "max_overlap": 0.75,  # Maximum allowed overlap fraction between ROIs before discarding one
            "high_pass": 100,  # Window size for running mean subtraction across time
            "smooth_masks": True,  # Whether to smooth ROI masks during final detection pass
            "max_iterations": 20,  # Maximum number of iterations for cell extraction
            "nbinned": 5000,  # Maximum number of binned frames to use for ROI detection
            "denoise": False,  # Whether to denoise the binned movie before cell detection

            # Cellpose Detection settings
            "anatomical_only": 0,  # If > 0, use Cellpose for anatomical detection (different modes 1-4)
            "diameter": 0,  # Diameter for Cellpose (if 0, diameter is estimated automatically)
            "cellprob_threshold": 0.0,  # Probability threshold for cell detection using Cellpose
            "flow_threshold": 1.5,  # Flow threshold used by Cellpose
            "spatial_hp_cp": 0,  # Window for spatial high-pass filtering before Cellpose processing
            "pretrained_model": "cyto",  # Path or type of the pretrained model to use with Cellpose

            # Signal extraction settings
            "neuropil_extract": True,  # Whether to extract the signal from the neuropil
            "allow_overlap": False,  # Whether to extract signals from overlapping ROIs
            "min_neuropil_pixels": 350,  # Minimum number of pixels required for neuropil extraction per cell
            "inner_neuropil_radius": 2,  # Number of pixels kept between the ROI and the neuropil donut
            "lam_percentile": 50,  # Percentile of lambda within area to ignore when excluding cell pixels

            # Spike deconvolution settings
            "spikedetect": True,  # Whether to perform spike deconvolution
            "neucoeff": 0.7,  # Neuropil coefficient applied to all ROIs
            "baseline": "maximin",  # Method to compute baseline ('maximin', 'constant', or 'constant_percentile')
            "win_baseline": 60.0,  # Window (in seconds) for the baseline computation filter
            "sig_baseline": 10.0,  # Gaussian filter width (in seconds) used in baseline computation
            "prctile_baseline": 8,  # Percentile of the trace to use as baseline (only for 'constant_percentile')

            # Classification settings
            "soma_crop": True,  # Whether to crop dendrites for cell classification statistics
            "use_builtin_classifier": False,
            # Whether to use the built-in classifier (overrides external classifier if True)
            "classifier_path": "",  # Path to the classifier file for cell classification

            # Channel 2 specific settings
            "chan2_thres": 0,  # Threshold for calling an ROI “detected” on a second channel

            # Miscellaneous settings
            "suite2p_version": ""  # Version of the suite2p pipeline used for these settings
        }

        # Ensure the file extension is .json
        if file_path.suffix.lower() != ".json":
            message = (
                f"Invalid file path provided. Expected a path ending with '.json', but got {file_path}."
            )
            console.error(message=message, error=ValueError)

        # Ensure that the output directory exists
        ensure_directory_exists(file_path)

        instance_data = asdict(self)

        merged_data = {**json_formatting, **instance_data}

        # Write the instance data to the JSON file with indentation for readability
        with open(file_path, "w") as json_file:
            json.dump(merged_data, json_file)

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



def main():
    input_file = Path("ops.json")
    output_file = Path(r"C:\Users\Kaze\Desktop\test.json")

    try:
        config = JsonConfig.from_json(input_file)
    except Exception as e:
        print(f"loading error：{e}")
        return

    try:
        config.to_json(output_file)
    except Exception as e:
        print(f"saving error：{e}")
        return

    print(f"successfully loaded from '{input_file}' and saved to '{output_file}'.")

if __name__ == "__main__":
    main()



