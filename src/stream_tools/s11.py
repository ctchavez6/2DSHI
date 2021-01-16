

def step_eleven(stream):
    print("Step 11 - Writing warp matrices to file")

    if stream.warp_matrix is not None:
        wm1_path = os.path.join(run_folder, 'wm1.npy')
        np.save(wm1_path, stream.warp_matrix)

    if stream.warp_matrix_2 is not None:
        wm2_path = os.path.join(run_folder, 'wm2.npy')
        np.save(wm2_path, stream.warp_matrix)