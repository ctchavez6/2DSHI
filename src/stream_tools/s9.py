import os
import csv as csv
from path_management import image_management as im
from image_processing import bit_depth_conversion as bdc


def step_nine(stream, start_writing_at, end_writing_at, run_folder, a_images, a_frames, b_prime_images, b_prime_frames,
              stats):
    r_matrices = stream.r_frames
    n_ = 0
    print("Writing R Matrices")
    for i in range(start_writing_at, end_writing_at + 1):
        n_ += 1
        r_matrix = r_matrices[i - 1]
        csv_path = os.path.join(run_folder, "r_matrix_{}.csv".format(n_))
        with open(csv_path, "w+", newline='') as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(r_matrix.tolist())

    n_ = 0
    a_frames_dir = os.path.join(run_folder, "cam_a_frames")
    print("Writing A Matrices")
    for i in range(start_writing_at, end_writing_at + 1):
        # for a_matrix in a_frames:
        n_ += 1
        a_matrix = a_frames[i - 1]
        csv_path = os.path.join(run_folder, "a_matrix_{}.csv".format(n_))
        with open(csv_path, "w+", newline='') as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(a_matrix.tolist())

        # a16 = bdc.to_16_bit(a_matrix)
        # im.save_img("a_{}.png".format(n_), a_frames_dir, a16)

    print("Writing A Images")
    n_ = 0
    for i in range(start_writing_at, end_writing_at + 1):
        # for img_a in a_images:
        n_ += 1
        img_a = a_images[i - 1]
        a16 = bdc.to_16_bit(img_a)
        im.save_img("a_{}.png".format(n_), a_frames_dir, a16)

    b_frames_dir = os.path.join(run_folder, "cam_b_frames")

    n_ = 0
    print("Writing B Matrices")
    for i in range(start_writing_at, end_writing_at + 1):
        # for b_matrix in b_prime_frames:
        n_ += 1
        b_matrix = b_prime_frames[i - 1]
        csv_path = os.path.join(run_folder, "b_matrix_{}.csv".format(n_))
        with open(csv_path, "w+", newline='') as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(b_matrix.tolist())

        # b16 = bdc.to_16_bit(b_matrix)
        # im.save_img("b_{}.png".format(n_), b_frames_dir, b16)

    print("Writing B Images")
    n_ = 0
    for i in range(start_writing_at, end_writing_at + 1):
        # for img_b in b_prime_images:
        n_ += 1
        img_b = b_prime_images[i - 1]
        b16 = bdc.to_16_bit(img_b)
        im.save_img("b_{}.png".format(n_), b_frames_dir, b16)

    print("Writing R Matrix Stats to file:")
    stats_csv_path = os.path.join(run_folder, "r_matrices_stats.csv")
    with open(stats_csv_path, "w+", newline='') as stats_csv:
        stats_csvWriter = csv.writer(stats_csv, delimiter=',')
        stats_csvWriter.writerow(stats[0])
        count = 0
        for i in range(start_writing_at, end_writing_at + 1):
            count += 1
            stats_csvWriter.writerow([count, stats[i][1], stats[i][2]])
    print("Matrices and Matrix Stats have finished writing to file.")
