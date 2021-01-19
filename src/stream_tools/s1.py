import cv2

y_n_msg = "Proceed? (y/n): "

def step_one(stream, histogram, continue_stream):
    start = input("Step 1 - Stream Raw Camera Feed -  {}".format(y_n_msg)).lower()
    display_stream = True if start == "y" else False

    if (stream.histocam_a is None or stream.histocam_b is None) and histogram:
        stream.histocam_a = stream.Histocam()
        stream.histocam_b = stream.Histocam()

    if display_stream:
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
