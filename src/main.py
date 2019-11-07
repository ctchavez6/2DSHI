"""Driver class - you should be able to run main from the command line along with the some command line arguments of
  your choice, and this script will call all other modules/scripts and feed those args/params as needed."""

import argparse
import os
import stream_cameras_and_histograms as streams

#import src.stream_cameras_and_histograms as stream

# Create an instance of an ArgumentParser Object
parser = argparse.ArgumentParser()



def validate_video_file(video_file_path_string):
    """
    Validates that the input video file exists and is in one of the accepted formats.

    Args:
        video_file_path_string: Path to video file.
    Raises:
        Exception: If invalid file or video.
    """

    if not os.path.exists(video_file_path_string):
        parser.error("\n\tThe file %s does not exist:" % video_file_path_string)

    acceptable_video_extensions = \
        [
            ".mov",
            ".avi",
            ".mp4"
        ]

    filename, file_extension = os.path.splitext(video_file_path_string)

    if file_extension not in acceptable_video_extensions:
        parser.error("\n\n\tPlease use one of the following video formats:" +
                     ''.join("\n\t\t%s" % x for x in acceptable_video_extensions) +
                     "\n\n\tYour input: %s\n" % video_file_path_string
                     )


def validate_color_input(color_input_string):
    """
    Validates user input for requested format for color (or lack ther eof).

    Args:
        color_input_string: Requested color format.
    Raises:
        Exception: If invalid or unaccepted color format.
    """

    acceptable_color_options = ["gray", "grayscale", "rgb"]
    if color_input_string not in acceptable_color_options:
        parser.error("\n\tPlease use one of the following color options:" +
                     ''.join("\n\t%s" % x for x in acceptable_color_options) +
                     "\n\tYour input: %s\n" % color_input_string
                     )


def validate_bins_input(bins_input_int):
    """
    Validates the user input for number of bins

    Args:
        bins_input_int: Requested number of bins.
    Raises:
        Exception: If requested number of bins outside a specified set of values.
    """
    acceptable_bins_options = [4096]
    if bins_input_int not in acceptable_bins_options:
        parser.error("\n\tPlease use one of the following options for number of bins:" +
                     ''.join("\n\t%s" % x for x in acceptable_bins_options) +
                     "\n\tYour input: %s\n" % bins_input_int
                     )


def validate_width_input(width_input_int):
    """
    Validates the user input for width.

    Args:
        width_input_int: Description
    Raises:
        Exception: If requested width outside a specified range.
    """

    min_width, max_width = 100, 1000

    if width_input_int not in range(min_width, max_width + 1):
        parser.error("\n\tPlease enter a width between %s and %s" % (min_width, max_width) +
                     "\n\tYour input: %s\n" % width_input_int
                     )


parser.add_argument('-f', '--file', default=None,
                    help='Path to video file (if not using camera)')
parser.add_argument('-c', '--color', type=str, default='gray',
                    help='Color space: "grayscale" (default) or "rgb"')
parser.add_argument('-b', '--bins', type=int, default=4096,
                    help='Number of bins per channel (default 4096)')
parser.add_argument('-w', '--width', type=int, default=0,
                    help='Resize video to specified width in pixels (maintains aspect ratio)')
parser.add_argument('-n', '--num_cameras', type=int, default=2,
                    help='Number of cameras (default 2)')


if __name__ == "__main__":

    print("Starting:")

    args = vars(parser.parse_args())  # Parse user arguments into a dictionary

    if isinstance(args['file'], type(None)):
        print("No file specified: Attempting video stream")
    else:
        validate_video_file(args['file'])  # Raises error if not a valid video

    validate_color_input(args['color'])

    validate_bins_input(args['bins'])

    if args['width'] is not 0:
        validate_width_input(args['width'])

    # Now that all the user inputs have be validated, we can begin running the script(s)
    parent_directory = os.path.dirname(os.getcwd())  # String representing parent directory of current working directory
    camera_configurations_folder = os.getcwd() + "\camera_configuration_files"
    camera_a_configuration = camera_configurations_folder + "\/23097552_setup_Oct23_padded_12.pfs"
    camera_b_configuration = camera_configurations_folder + "\/23097552_setup_Oct23_padded_12.pfs"

    config_files_by_cam = dict()
    config_files_by_cam["a"] = camera_a_configuration
    config_files_by_cam["b"] = camera_b_configuration

    devices_found, tlFactory_found = streams.find_devices()

    cameras = streams.get_cameras(
        devices=devices_found,
        tlFactory = tlFactory_found,
        config_files=config_files_by_cam,
        num_cameras=args["num_cameras"])

    figs, histograms, lines = streams.initialize_histograms(args['bins'])
    streams.stream_cam_to_histograms(cams_dict=cameras, figures=figs, histograms_dict=histograms, lines=lines, bins=4096)