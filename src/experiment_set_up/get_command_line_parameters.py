import argparse

def initialize_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-rpp', '--RunPreviousParameters', default=0, action='store_true',
                        help="Show a grid on the images.")
    parser.add_argument('-va', '--video_a', default=None,
                        help='Path to video file for Camera A (if not using camera)')
    parser.add_argument('-vb', '--video_b', default=None,
                        help='Path to video file for Camera B (if not using camera)')
    parser.add_argument('-ccf', '--camera_configuration_file', default=None,
                        help='Desired camera configuration file (if other than default_camera_configuration.pfs)')
    parser.add_argument('-ccfa', '--camera_configuration_file_a', default=None,
                        help='Desired camera configuration file for A (if other than default_camera_configuration.pfs)')
    parser.add_argument('-ccfb', '--camera_configuration_file_b', default=None,
                        help='Desired camera configuration file for B (if other than default_camera_configuration.pfs)')
    parser.add_argument('-e', '--ExposureTime', type=str, default=None,
                        help='Exposure time in us (microseconds) [overwrites value in default_camera_configuration.pfs]')
    parser.add_argument('-r', '--AcquisitionFrameRate', type=str, default=None,
                        help='Acquisition frame rate (Hz) [overwrites value in default_camera_configuration.pfs]')
    parser.add_argument('-si', '--SaveImages', type=int, default=0,
                        help="1 if you'd like to save the images from this run, else 0.")
    parser.add_argument('-sv', '--SaveVideos', type=int, default=0,
                        help="1 if you'd like to save the videos from this run, else 0.")
    parser.add_argument('-fb', '--FrameBreak', type=int, default=-1,
                        help="How many frames you'd like to grab/save.")
    parser.add_argument('-dhc', '--DisplayHistocam', type=int, default=0,
                        help="How many frames you'd like to grab/save.")
    parser.add_argument('-shc', '--SaveHistocam', type=int, default=0,
                        help="How many frames you'd like to grab/save.")
    parser.add_argument('-raw', '--Raw', type=int, default=1,
                        help="Show raw image data.")
    parser.add_argument('-rf', '--ResizeFactor', type=float, default=1.0,
                        help="Show raw image data.")
    parser.add_argument('-g', '--Grid', type=int, default=0,
                        help="Show a grid on the images.")
    parser.add_argument('-c1t', '--CrystalTemp1', type=float, default=0,
                        help="Show a grid on the images.")
    parser.add_argument('-c2t', '--CrystalTemp2', type=float, default=0,
                        help="Show a grid on the images.")
    parser.add_argument('-c_ang', '--CompensatorAngle', type=float, default=0,
                        help="Show a grid on the images.")
    parser.add_argument('-tar', '--Target', type=int, default=0,
                        help="Show a grid on the images.")

    return parser
