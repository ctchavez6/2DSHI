import cv2
from image_processing import bit_depth_conversion as bdc
import json


y_n_msg = "Proceed? (y/n): "
eight_bit_max = (2 ** 8) - 1

def step_three(stream, continue_stream):
    find_centers_ = input("Step 3 - Find Brightest Pixel Locations - {}".format(y_n_msg))

    if find_centers_.lower() == "y":
        continue_stream = True

    while continue_stream:
        stream.frame_count += 1
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames(warp_matrix=stream.warp_matrix)

        a_as_16bit = bdc.to_16_bit(stream.current_frame_a)
        b_as_16bit = bdc.to_16_bit(stream.current_frame_b)
        max_pixel_a, max_pixel_b = stream.find_centers(a_as_16bit, b_as_16bit)

        stream.max_pixel_a = max_pixel_a
        stream.max_pixel_b = max_pixel_b

        a_as_16bit = cv2.circle(a_as_16bit, max_pixel_a, 10, (0, eight_bit_max, 0), 2)
        b_as_16bit = cv2.circle(b_as_16bit, max_pixel_b, 10, (0, eight_bit_max, 0), 2)

        cv2.imshow("A", a_as_16bit)
        cv2.imshow("B Prime", b_as_16bit)
        continue_stream = stream.keep_streaming()



    cv2.destroyAllWindows()

