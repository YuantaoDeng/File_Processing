import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from dacite import from_dict, Config
from typing import Any, List, Dict
import numpy as np
from ataraxis_base_utilities import console, ensure_directory_exists




@dataclass
class UniConfig:
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

    def to_npy(self, file_path: Path) -> None:
        """
        Converts the dataclass instance to a dictionary and saves it as an .npy file.

        Args:
            file_path: The path to the NPY file to write.
                       The file path must end with ".npy". If the file does not exist, the directory
                       will be created; if it exists, it will be overwritten.

        Raises:
            ValueError: If the file path does not have a '.npy' extension.
        """
        if file_path.suffix.lower() != ".npy":
            message = (
                f"Invalid file path provided. Expected a path ending with '.npy', but got {file_path}."
            )
            console.error(message=message, error=ValueError)

        # Ensure that the output directory exists
        ensure_directory_exists(file_path)

        # Convert the dataclass instance to a dictionary
        instance_data = asdict(self)

        # Save the dictionary to an .npy file (using pickle under the hood)
        np.save(file_path, instance_data)

    @classmethod
    def from_npy(cls, file_path: Path) -> "NpyConfig":
        """
        Instantiates the class using data loaded from the provided .npy file.

        This method reads the NPY file, converts it to a dictionary, and then uses dacite
        to instantiate the class. It disables dacite's built-in type checking, so you may need to
        add extra validation if required.

        Args:
            file_path: The path to the NPY file to read.

        Returns:
            A new dataclass instance created using the data read from the NPY file.

        Raises:
            ValueError: If the file path does not have a '.npy' extension.
        """
        if file_path.suffix.lower() != ".npy":
            message = (
                f"Invalid file path provided. Expected a path ending with '.npy', but got {file_path}."
            )
            console.error(message=message, error=ValueError)

        # Load the data from the .npy file.
        loaded = np.load(file_path, allow_pickle=True)
        # If the loaded object is a numpy array (as is typical), extract the original dictionary.
        if isinstance(loaded, np.ndarray):
            loaded = loaded.item()

        config_dict: dict[Any, Any] = dict(loaded)

        # Disable dacite's built-in type checking.
        class_config = Config(check_types=False)

        # Instantiate the dataclass using dacite.
        instance = from_dict(data_class=cls, data=config_dict, config=class_config)
        return instance


@dataclass
class MyConfig:
    """
    A configuration class with a hierarchical structure using nested classes.
    You can access configuration items like:
        config.main.nplanes
        config.file_io.save_folder
    """

    @dataclass
    class Main:
        nplanes: int = 1  # Each tiff has this many planes in sequence.
        nchannels: int = 1  # Each tiff has this many channels per plane.
        functional_chan: int = 1  # This channel is used to extract functional ROIs (1-based, so 1 means first channel, and 2 means second channel).
        tau: float = 1.0  # The timescale of the sensor (in seconds), used for deconvolution kernel.
        force_sktiff: bool = False  # Specifies whether or not to use scikit-image for reading in tiffs.
        fs: float = 10.0  # Sampling rate (per plane).
        do_bidiphase: bool = False  # Whether or not to compute bidirectional phase offset from misaligned line scanning experiment (applies to 2P recordings only).
        bidiphase: int = 0  # bidirectional phase offset from line scanning (set by user).
        bidi_corrected: bool = False  # Specifies whether to do bidi correction.
        frames_include: int = -1  # If greater than zero, only frames_include frames are processed.
        multiplane_parallel: bool = False  # Specifies whether or not to run pipeline on server.
        ignore_flyback: List[int] = field(default_factory=list)  # Specifies which planes will be ignored as flyback planes by the pipeline.

    @dataclass
    class FileIO:
        fast_disk: List[str] = field(default_factory=list)  # Specifies location where temporary binary file will be stored.
        delete_bin: bool = False  # Specifies whether to delete binary file created during registration stage.
        mesoscan: bool = False  # Specifies whether file being read in is a scanimage mesoscope recording.
        bruker: bool = False  # Specifies whether provided tif files are single page BRUKER tiffs.
        bruker_bidirectional: bool = False  # Specifies whether BRUKER files are bidirectional multiplane recordings.
        h5py: List[str] = field(default_factory=list)  # Specifies path to h5py file that will be used as inputs.
        h5py_key: str = "data"  # Key used to access data array in h5py file.
        nwb_file: str = ""  # Specifies path to NWB file you use to use as input.
        nwb_driver: str = ""  # Location of driver for NWB file.
        nwb_series: str = ""  # Name of TwoPhotonSeries values you wish to retrieve from your NWB file.
        save_path0: List[str] = field(default_factory=list)  # List containing pathname of where you’d like to save your pipeline results.
        save_folder: List[str] = field(default_factory=list)  # List containing directory name you’d like results to be saved under.
        look_one_level_down: bool = False  # Specifies whether to look in all subfolders when searching for tiffs.
        subfolders: List[str] = field(default_factory=list)  # Specifies subfolders you’d like to look through.
        move_bin: bool = False  #  If True and ops['fast_disk'] is different from ops[save_disk], the created binary file is moved to ops['save_disk'].

    @dataclass
    class Output:
        preclassify: float = 0.0  # Apply classifier before signal extraction with probability threshold of “preclassify”.
        save_nwb: bool = False  # Whether to save output as NWB file.
        save_mat: bool = False  # Whether to save the results in matlab format in file “Fall.mat”.
        combined: bool = True  # Combine results across planes in separate folder “combined” at end of processing.
        aspect: float = 1.0  # Ratio of um/pixels in X to um/pixels in Y (ONLY for correct aspect ratio in GUI, not used for other processing).
        report_time: bool = True  #  whether or not to return a timing dictionary for each plane.

    @dataclass
    class Registration:
        do_registration: bool = True  # Whether or not to run registration.
        align_by_chan: int = 1  # Which channel to use for alignment (1-based, so 1 means 1st channel and 2 means 2nd channel).
        nimg_init: int = 300  # How many frames to use to compute reference image for registration.
        batch_size: int = 500  # How many frames to register simultaneously in each batch.
        maxregshift: float = 0.1  # The maximum shift as a fraction of the frame size.
        smooth_sigma: float = 1.15  # Standard deviation in pixels of the gaussian used to smooth the phase correlation between the reference image and the frame which is being registered.
        smooth_sigma_time: float = 0.0  # Standard deviation in time frames of the gaussian used to smooth the data before phase correlation is computed.
        keep_movie_raw: bool = False  # Whether or not to keep the binary file of the non-registered frames.
        two_step_registration: bool = False  # Whether or not to run registration twice (for low SNR data).
        reg_tif: bool = False  # Whether or not to write the registered binary to tiff files.
        reg_tif_chan2: bool = False  # Whether or not to write the registered binary of the non-functional channel to tiff files.
        subpixel: int = 10  # Precision of Subpixel Registration (1/subpixel steps).
        th_badframes: float = 1.0  # Involved with setting threshold for excluding frames for cropping.
        norm_frames: bool = True  # Normalize frames when detecting shifts.
        force_refImg: bool = False  # Specifies whether to use refImg stored in ops.
        pad_fft: bool = False  # Specifies whether to pad image or not during FFT portion of registration.

    @dataclass
    class OnePRegistration:
        one_p_reg: bool = False  # Whether to perform high-pass spatial filtering and tapering.
        spatial_hp_reg: int = 42  # Window in pixels for spatial high-pass filtering before registration.
        pre_smooth: float = 0.0  # If > 0, defines stddev of Gaussian smoothing, which is applied before spatial high-pass filtering.
        spatial_taper: float = 40.0  # How many pixels to ignore on edges.

    @dataclass
    class NonRigid:
        nonrigid: bool = True  # Whether or not to perform non-rigid registration.
        block_size: List[int] = field(default_factory=lambda: [128, 128])  # Size of blocks for non-rigid registration, in pixels.
        snr_thresh: float = 1.2  # How big the phase correlation peak has to be relative to the noise in the phase correlation map for the block shift to be accepted.
        maxregshiftNR: float = 5.0  # Maximum shift in pixels of a block relative to the rigid shift.

    @dataclass
    class ROIDetection:
        roidetect: bool = True  # Whether or not to run ROI detect and extraction.
        sparse_mode: bool = True  # Whether or not to use sparse_mode cell detection.
        spatial_scale: int = 0  # What the optimal scale of the recording is in pixels.
        connected: bool = True  # Whether or not to require ROIs to be fully connected.
        threshold_scaling: float = 1.0  # This controls the threshold at which to detect ROIs (how much the ROIs have to stand out from the noise to be detected).
        spatial_hp_detect: int = 25  # Window for spatial high-pass filtering for neuropil subtracation before ROI detection takes place.
        max_overlap: float = 0.75  # After detection, ROIs with more than ops[‘max_overlap’] fraction of their pixels overlapping with other ROIs will be discarded.
        high_pass: int = 100  # Running mean subtraction across time with window of size ‘high_pass’.
        smooth_masks: bool = True  # Whether to smooth masks in final pass of cell detection.
        max_iterations: int = 20  # How many iterations over which to extract cells - at most ops[‘max_iterations’].
        nbinned: int = 5000  # Maximum number of binned frames to use for ROI detection.
        denoise: bool = False  #  Whether or not binned movie should be denoised before cell detection in sparse_mode.

    @dataclass
    class CellposeDetection:
        anatomical_only: int = 0  # Whether or not binned movie should be denoised before cell detection in sparse_mode.
                                  # 1: Will find masks on max projection image divided by mean image.
                                  # 2: Will find masks on mean image.
                                  # 3: Will find masks on enhanced mean image.
                                  # 4: Will find masks on maximum projection image.
        diameter: int = 0  # Diameter that will be used for cellpose.
        cellprob_threshold: float = 0.0  # Specifies threshold for cell detection that will be used by cellpose.
        flow_threshold: float = 1.5  # Specifies flow threshold that will be used for cellpose.
        spatial_hp_cp: int = 0  # Window for spatial high-pass filtering of image to be used for cellpose.
        pretrained_model: str = "cyto"  # Path to pretrained model or string for model type (can be user’s model ).

    @dataclass
    class SignalExtraction:
        neuropil_extract: bool = True  # Whether or not to extract signal from neuropil.
        allow_overlap: bool = False  # Whether or not to extract signals from pixels which belong to two ROIs.
        min_neuropil_pixels: int = 350  # Minimum number of pixels used to compute neuropil for each cell.
        inner_neuropil_radius: int = 2  # Number of pixels to keep between ROI and neuropil donut.
        lam_percentile: int = 50  # Percentile of Lambda within area to ignore when excluding cell pixels for neuropil extraction.

    @dataclass
    class SpikeDeconvolution:
        spikedetect: bool = True  # Whether or not to run spike_deconvolution.
        neucoeff: float = 0.7  # Neuropil coefficient for all ROIs.
        baseline: str = "maximin"  # How to compute the baseline of each trace.
        win_baseline: float = 60.0  # Window for maximin filter in seconds.
        sig_baseline: float = 10.0  # Gaussian filter width in seconds.
        prctile_baseline: int = 8  # Percentile of trace to use as baseline.

    @dataclass
    class Classification:
        soma_crop: bool = True  # Specifies whether to crop dendrites for cell classification stats.
        use_builtin_classifier: bool = False  # Specifies whether or not to use built-in classifier for cell detection.
        classifier_path: str = ""  #  Path to classifier file you want to use for cell classification.

    @dataclass
    class Channel2:
        chan2_thres: int = 0  # Threshold for calling an ROI “detected” on a second channel.

    @dataclass
    class Miscellaneous:
        suite2p_version: str = ""  # Specifies version of suite2p pipeline that was run with these settings.

    # Nested class for extra configuration parameters.
    # This field is intended to store extra subfields that exist in the input file
    # but are not defined in the default MyConfig schema.
    @dataclass
    class Extra:
        fields: Dict[str, Any] = field(default_factory=dict)

    # Define the instances of each nested settings class as fields
    main: Main = field(default_factory=Main)
    file_io: FileIO = field(default_factory=FileIO)
    output: Output = field(default_factory=Output)
    registration: Registration = field(default_factory=Registration)
    one_p_registration: OnePRegistration = field(default_factory=OnePRegistration)
    non_rigid: NonRigid = field(default_factory=NonRigid)
    roi_detection: ROIDetection = field(default_factory=ROIDetection)
    cellpose_detection: CellposeDetection = field(default_factory=CellposeDetection)
    signal_extraction: SignalExtraction = field(default_factory=SignalExtraction)
    spike_deconvolution: SpikeDeconvolution = field(default_factory=SpikeDeconvolution)
    classification: Classification = field(default_factory=Classification)
    channel2: Channel2 = field(default_factory=Channel2)
    miscellaneous: Miscellaneous = field(default_factory=Miscellaneous)
    extra: Extra = field(default_factory=Extra)



# Tset Example
def main():
    input_file = Path("")
    output_file = Path("")

    try:
        config = MyConfig.from_json(input_file)
    except Exception as e:
        print(f"loading error：{e}")
        return

    config.main.nplanes = config.nplanes + 1

    try:
        config.to_json(output_file)
    except Exception as e:
        print(f"saving error：{e}")
        return

    print(f"successfully loaded from '{input_file}' and saved to '{output_file}'.")

if __name__ == "__main__":
    main()



