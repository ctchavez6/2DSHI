import cv2


def step_one(stream, histogram, display_stream, continue_stream):

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
