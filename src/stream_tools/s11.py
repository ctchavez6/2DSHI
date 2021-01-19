import numpy as np
import os
import pickle


def step_eleven(stream, run_folder):
    print("Step 11: Writing Calibration Parameters to file")

    if stream.warp_matrix is not None or stream.warp_matrix_2 is not None:
        print("\tWriting warp matrices to file")
        if stream.warp_matrix is not None:
            wm1_path = os.path.join(run_folder, 'wm1.npy')
            np.save(wm1_path, stream.warp_matrix)

        if stream.warp_matrix_2 is not None:
            wm2_path = os.path.join(run_folder, 'wm2.npy')
            np.save(wm2_path, stream.warp_matrix)

    if stream.max_pixel_a or stream.max_pixel_b:
        print("\tWriting brightest pixels to file")
        if stream.max_pixel_a is not None:
            with open(os.path.join(run_folder, 'max_pixel_a.p'), 'wb') as fp:
                pickle.dump(stream.max_pixel_a, fp, protocol=pickle.HIGHEST_PROTOCOL)

        if stream.max_pixel_b is not None:
            with open(os.path.join(run_folder, 'max_pixel_b.p'), 'wb') as fp:
                pickle.dump(stream.max_pixel_b, fp, protocol=pickle.HIGHEST_PROTOCOL)

    if stream.static_center_a is not None or stream.static_center_b is not None:
        print("\tWriting static centers to file")
        if stream.static_center_a is not None:
            with open(os.path.join(run_folder, 'static_center_a.p'), 'wb') as fp:
                pickle.dump(stream.static_center_a, fp, protocol=pickle.HIGHEST_PROTOCOL)
        if stream.static_center_b is not None:
            with open(os.path.join(run_folder, 'static_center_b.p'), 'wb') as fp:
                pickle.dump(stream.static_center_b, fp, protocol=pickle.HIGHEST_PROTOCOL)

    if stream.static_sigmas_x is not None or stream.static_sigmas_y is not None:
        print("\tWriting static sigmas to file")
        if stream.static_sigmas_x is not None:
            with open(os.path.join(run_folder, 'static_sigma_x.p'), 'wb') as fp:
                pickle.dump(stream.static_sigmas_x, fp, protocol=pickle.HIGHEST_PROTOCOL)
        if stream.static_sigmas_y is not None:
            with open(os.path.join(run_folder, 'static_sigma_y.p'), 'wb') as fp:
                pickle.dump(stream.static_sigmas_y, fp, protocol=pickle.HIGHEST_PROTOCOL)