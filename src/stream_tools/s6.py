from image_processing import bit_depth_conversion as bdc
import cv2


y_n_msg = "Proceed? (y/n): "


def step_six_a(stream, continue_stream):
    close_in = input("Step 6A - Close in on ROI - {}".format(y_n_msg))

    if close_in.lower() == "y":
        continue_stream = True

    while continue_stream:
        stream.frame_count += 1
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames(warp_matrix=stream.warp_matrix)

        x_a, y_a = stream.static_center_a
        x_b, y_b = stream.static_center_b

        n_sigma = 1

        stream.roi_a = stream.current_frame_a[
                     y_a - n_sigma * stream.static_sigmas_y: y_a + n_sigma * stream.static_sigmas_y + 1,
                     x_a - n_sigma * stream.static_sigmas_x: x_a + n_sigma * stream.static_sigmas_x + 1
                     ]

        stream.roi_b = stream.current_frame_b[
                     y_b - n_sigma * stream.static_sigmas_y: y_b + n_sigma * stream.static_sigmas_y + 1,
                     x_b - n_sigma * stream.static_sigmas_x: x_b + n_sigma * stream.static_sigmas_x + 1
                     ]

        cv2.imshow("ROI A", bdc.to_16_bit(stream.roi_a))
        cv2.imshow("ROI B Prime", bdc.to_16_bit(stream.roi_b))
        continue_stream = stream.keep_streaming()

    cv2.destroyAllWindows()