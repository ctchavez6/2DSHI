import matplotlib.pyplot as plt
import numpy as np
import path_management.directory_management as dirs
import path_management.image_management as img_tools
import cv2
import os


run = "2019_12_14__13_08"

img_a_path = dirs.get_latest_run()

algebra_directory = os.path.join(img_a_path, "Image_Algebra")
img_a_path = os.path.join(img_a_path, "cam_a_frames")
img_a_path = os.path.join(img_a_path, "cam_a_frame_1.png")
img_a = img_tools.read_img(img_a_path)
img_a_12_bit = img_tools.reduce_bit_depth(image_array=img_a, original_bit_depth=16, intended_bit_depth=12)

img_b_prime_path = os.path.join(algebra_directory, "B_Prime.png")
img_b_prime = img_tools.read_img(img_b_prime_path)
img_b_prime_12_bit = img_tools.reduce_bit_depth(image_array=img_b_prime, original_bit_depth=16, intended_bit_depth=12)


subtracted = np.subtract(img_a_12_bit, img_b_prime_12_bit)
added = np.add(img_a_12_bit, img_b_prime_12_bit)


def create_colormap(file_name, save_directory, array):
    save_path = os.path.join(save_directory, file_name)
    fig = plt.figure()
    imgplot = plt.imshow(array)
    imgplot.set_cmap('bwr')

    for im in plt.gca().get_images():
        im.set_clim(-4095 * 2, 4095 * 2)

    plt.colorbar()
    plt.show()
    fig.savefig(save_path)



create_colormap("A_Minus_B_Prime_Colormap.png", algebra_directory, subtracted)
create_colormap("A_Plus_B_Prime_Colormap.png", algebra_directory, added)
