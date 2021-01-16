from coregistration import find_gaussian_profile as fgp
from image_processing import bit_depth_conversion as bdc
import cv2


y_n_msg = "Proceed? (y/n): "

def step_five(stream, continue_stream):
    find_rois_ = input("Step 5 - Define Regions of Interest - {}".format(y_n_msg))

    if find_rois_.lower() == "y":
        continue_stream = True

    while continue_stream:
        stream.frame_count += 1
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames(warp_matrix=stream.warp_matrix)

        try:
            for img_12bit in [stream.current_frame_a]:
                center_ = stream.static_center_a

                n_sigma = 1

                stream.mu_x, stream.sigma_x_a, stream.amp_x = fgp.get_gaus_boundaries_x(img_12bit, center_)
                stream.mu_y, stream.sigma_y_a, stream.amp_y = fgp.get_gaus_boundaries_y(img_12bit, center_)

                img_12bit[:, int(center_[0]) + int(stream.sigma_x_a * n_sigma)] = 4095
                img_12bit[:, int(center_[0]) - int(stream.sigma_x_a * n_sigma)] = 4095

                img_12bit[int(center_[1]) + int(stream.sigma_y_a * n_sigma), :] = 4095
                img_12bit[int(center_[1]) - int(stream.sigma_y_a * n_sigma), :] = 4095

                if stream.frame_count % 10 == 0:
                    print("\tA  - Sigma X, Sigma Y - {}".format((int(stream.sigma_x_a), int(stream.sigma_y_a))))

            for img_12bit in [stream.current_frame_b]:
                center_ = stream.static_center_b

                stream.mu_x, stream.sigma_x_b, stream.amp_x = fgp.get_gaus_boundaries_x(img_12bit, center_)
                stream.mu_y, stream.sigma_y_b, stream.amp_y = fgp.get_gaus_boundaries_y(img_12bit, center_)

                img_12bit[:, int(center_[0]) + int(stream.sigma_x_b * n_sigma)] = 4095
                img_12bit[:, int(center_[0]) - int(stream.sigma_x_b * n_sigma)] = 4095

                img_12bit[int(center_[1]) + int(stream.sigma_y_b * n_sigma), :] = 4095
                img_12bit[int(center_[1]) - int(stream.sigma_y_b * n_sigma), :] = 4095

                if stream.frame_count % 10 == 0:
                    print("\tB  - Sigma X, Sigma Y - {}".format((int(stream.sigma_x_a), int(stream.sigma_y_a))))

            a_as_16bit = bdc.to_16_bit(stream.current_frame_a)
            b_as_16bit = bdc.to_16_bit(stream.current_frame_b)

            cv2.imshow("A", a_as_16bit)
            cv2.imshow("B Prime", b_as_16bit)

        except Exception:
            print("Exception Occurred")

        continue_stream = stream.keep_streaming()

        if continue_stream is False:
            stream.static_sigmas_x = int(max(stream.sigma_a_x, stream.sigma_b_x))
            stream.static_sigmas_y = int(max(stream.sigma_a_y, stream.sigma_b_y))

    cv2.destroyAllWindows()