import cv2
from experiment_set_up import user_input_validation as uiv

def step_one(stream, histogram, continue_stream):
    step_description = "Step 1 - Stream Raw Camera Feed"
    start = uiv.yes_no_quit(step_description)
    display_stream = True if start is True else False

    if (stream.histocam_a is None or stream.histocam_b is None) and histogram:
        stream.histocam_a = stream.Histocam()
        stream.histocam_b = stream.Histocam()

    if display_stream is True:
        continue_stream = True
    else:
        stream.frame_count += 1
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames()
        stream.pre_alignment(histogram)

    while continue_stream:
        stream.frame_count += 1
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames()
        stream.pre_alignment(histogram)
        continue_stream = stream.keep_streaming()

    cv2.destroyAllWindows()
