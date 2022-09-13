import cv2
from stream_tools.curveDigitizer import run as runCurve
from experiment_set_up import user_input_validation as uiv
from constants import STEP_DESCRIPTIONS as sd

def step_one(stream, histogram, continue_stream):
    """
    This step asks for a delineator for the run

    Args:
        stream (Stream): An instance of the Stream Class
        histogram (bool): True if you'd like to display histograms for pixel values, false otherwise.
        continue_stream (bool): True to continue streaming camera feeds, false otherwise

    Returns:
        bool: The return value. True for success, False otherwise.
    """
    step_description = sd.S00_DESC.value
    start = uiv.yes_no_quit(step_description)  # Grabs user input for whether or not they want to proceed w/ Step 1.
    display_stream = True if start is True else False

    if (stream.histocam_a is None or stream.histocam_b is None) and histogram:  # If use wants to display Histograms
        stream.histocam_a = stream.Histocam()
        stream.histocam_b = stream.Histocam()

    if display_stream is True:
        continue_stream = True
    else:  # We need to at the very least grab some initial frames even if we don't display them.
        stream.frame_count += 1
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames()
        stream.pre_alignment(histogram)

    while continue_stream:
        stream.frame_count += 1
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames()
        stream.pre_alignment(histogram)
        continue_stream = stream.keep_streaming()

    cv2.destroyAllWindows()  # Upon leaving stream, we want to close any CV2 windows.

    # testing the curve digitizer
    # coordsListA, coordsListB = runCurve(stream)

