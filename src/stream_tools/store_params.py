import numpy as np
import os
import pickle

def store_warp_matrices(stream, run_folder):
    """
    Args:
        stream (Stream): The Stream object we are pulling the warp matrices from.
        run_folder (str): The string path to the run folder to which we are storing the warp matrices.

    """
    if stream.warp_matrix is not None or stream.warp_matrix_2 is not None:
        print("\tWriting warp matrices to file")
        if stream.warp_matrix is not None:
            wm1_path = os.path.join(run_folder, 'wm1.npy')
            np.save(wm1_path, stream.warp_matrix)

        if stream.warp_matrix_2 is not None:
            wm2_path = os.path.join(run_folder, 'wm2.npy')
            np.save(wm2_path, stream.warp_matrix)


def store_brightest_pixels(stream, run_folder):
    """

    Args:
        stream (Stream): The Stream object we are pulling the brightest pixel  from.
        run_folder (str): The string path to the run folder to which we are storing the warp matrices.

    """
    if stream.max_pixel_a or stream.max_pixel_b:
        print("\tWriting brightest pixels to file")
        if stream.max_pixel_a is not None:
            with open(os.path.join(run_folder, 'max_pixel_a.p'), 'wb') as fp:
                pickle.dump(stream.max_pixel_a, fp, protocol=pickle.HIGHEST_PROTOCOL)

        if stream.max_pixel_b is not None:
            with open(os.path.join(run_folder, 'max_pixel_b.p'), 'wb') as fp:
                pickle.dump(stream.max_pixel_b, fp, protocol=pickle.HIGHEST_PROTOCOL)


def store_static_centers(stream, run_folder):
    if stream.static_center_a is not None or stream.static_center_b is not None:
        print("\tWriting static centers to file")
        if stream.static_center_a is not None:
            with open(os.path.join(run_folder, 'static_center_a.p'), 'wb') as fp:
                pickle.dump(stream.static_center_a, fp, protocol=pickle.HIGHEST_PROTOCOL)
        if stream.static_center_b is not None:
            with open(os.path.join(run_folder, 'static_center_b.p'), 'wb') as fp:
                pickle.dump(stream.static_center_b, fp, protocol=pickle.HIGHEST_PROTOCOL)

def store_max_n_sigma(stream, run_folder):
    if stream.max_n_sigma is not None:
        print("\tWriting max n_sigma to file")
        with open(os.path.join(run_folder, 'max_n_sigma.p'), 'wb') as fp:
            pickle.dump(stream.max_n_sigma, fp, protocol=pickle.HIGHEST_PROTOCOL)


def store_static_sigmas(stream, run_folder):
    if stream.static_sigmas_x is not None or stream.static_sigmas_y is not None:
        print("\tWriting static sigmas to file")
        if stream.static_sigmas_x is not None:
            with open(os.path.join(run_folder, 'static_sigma_x.p'), 'wb') as fp:
                pickle.dump(stream.static_sigmas_x, fp, protocol=pickle.HIGHEST_PROTOCOL)
        if stream.static_sigmas_y is not None:
            with open(os.path.join(run_folder, 'static_sigma_y.p'), 'wb') as fp:
                pickle.dump(stream.static_sigmas_y, fp, protocol=pickle.HIGHEST_PROTOCOL)


def store_offsets(stream, run_folder):
    if stream.h_offset is not None:
        with open(os.path.join(run_folder, 'h_offset.p'), 'wb') as fp:
            print("\tWriting h_offset to file")
            pickle.dump(stream.h_offset, fp, protocol=pickle.HIGHEST_PROTOCOL)
    if stream.v_offset is not None:
        with open(os.path.join(run_folder, 'v_offset.p'), 'wb') as fp:
            print("\tWriting v_offset to file")
            pickle.dump(stream.v_offset, fp, protocol=pickle.HIGHEST_PROTOCOL)


def store_n_sigma(stream, run_folder):
    if stream.n_sigma is not None:
        print("\tWriting n_sigma to file")
        with open(os.path.join(run_folder, 'n_sigma.p'), 'wb') as fp:
            pickle.dump(stream.n_sigma, fp, protocol=pickle.HIGHEST_PROTOCOL)
