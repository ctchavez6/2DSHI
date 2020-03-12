import sys
import os
import cv2
sys.path.append("..")


from src.image_processing import img_algebra as ia
from src.path_management import image_management as im
from src.image_processing import create_algebra_colormaps as cac
import numpy as np


EPSILON = sys.float_info.epsilon  # Smallest possible difference.


def convert_to_rgb_pos(val):
    minval, maxval = 0, 2*((2**12) - 1)
    colors = [(sixteen_bit_max, sixteen_bit_max, sixteen_bit_max), (0, 0, sixteen_bit_max)]  # [WHITE, RED]

    # "colors" is a series of RGB colors delineating a series of
    # adjacent linear color gradients between each pair.
    # Determine where the given value falls proportionality within
    # the range from minval->maxval and scale that fractional value
    # by the total number in the "colors" pallette.
    #print("val\n", val)
    i_f = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    # Determine the lower index of the pair of color indices this
    # value corresponds and its fractional distance between the lower
    # and the upper colors.
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.
    # Does it fall exactly on one of the color points?
    if f < EPSILON:
        return colors[i]
    else:  # Otherwise return a color within the range between them.
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))






def convert_to_rgb_neg(val):
    minval, maxval = -1*((2**12) - 1), 0
    colors = [(sixteen_bit_max, 0, 0), (sixteen_bit_max, sixteen_bit_max, sixteen_bit_max)]  # [WHITE, RED]

    # "colors" is a series of RGB colors delineating a series of
    # adjacent linear color gradients between each pair.
    # Determine where the given value falls proportionality within
    # the range from minval->maxval and scale that fractional value
    # by the total number in the "colors" pallette.
    i_f = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    # Determine the lower index of the pair of color indices this
    # value corresponds and its fractional distance between the lower
    # and the upper colors.
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.
    # Does it fall exactly on one of the color points?
    if f < EPSILON:
        return colors[i]
    else:  # Otherwise return a color within the range between them.
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))






def convert_to_rgb_all(val):
    minval, maxval = -2*((2**12) - 1), 2*((2**12) - 1)
    colors = [(sixteen_bit_max, 0, 0), (sixteen_bit_max, sixteen_bit_max, sixteen_bit_max), (0, 0, sixteen_bit_max)]  # [Blue, WHITE, RED]

    # "colors" is a series of RGB colors delineating a series of
    # adjacent linear color gradients between each pair.
    # Determine where the given value falls proportionality within
    # the range from minval->maxval and scale that fractional value
    # by the total number in the "colors" pallette.
    i_f = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    # Determine the lower index of the pair of color indices this
    # value corresponds and its fractional distance between the lower
    # and the upper colors.
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.
    # Does it fall exactly on one of the color points?
    if f < EPSILON:
        return colors[i]
    else:  # Otherwise return a color within the range between them.
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))


#from colour_system import cs_hdtv

if __name__ == "__main__":
    print("Welcome to the 2DSHI Software Package tests.\n"
          "Please make sure you are running this script from your src directory.\n")

    a_func = np.vectorize(convert_to_rgb_pos)
    s_func = np.vectorize(convert_to_rgb_neg)
    func = np.vectorize(convert_to_rgb_all)
    sixteen_bit_max = (2**16) - 1



    #b, g, r = convert_to_rgb_pos(val)

    #b, g, r = convert_to_rgb_neg(val)

    test_materials = os.path.join(os.getcwd(), "tests")
    test_materials = os.path.join(test_materials, "2020_01_04__15_56")
    cam_a_frames = os.path.join(test_materials, "cam_a_frames")
    cam_b_frames = os.path.join(test_materials, "cam_b_frames")

    img_a = np.asarray(im.read_img(os.path.join(cam_a_frames, "a_1.png")), dtype='uint16')
    img_b = np.asarray(im.read_img(os.path.join(cam_b_frames, "b_1.png")), dtype='uint16')
    img_b_prime = np.asarray(im.read_img(os.path.join(test_materials, "B_Prime.png")), dtype='uint16')

    print("In Image A: Range of pixel intensity is {} to {}".format(np.min(img_a)/16, np.max(img_a)/16))
    print("In Image B': Range of pixel intensity is {} to {}".format(np.min(img_b_prime)/16, np.max(img_b_prime)/16))

    added = ia.add_imgs(img_a/16, img_b_prime/16)
    subtracted = np.add(img_a/16, (-1)*(img_b_prime*(1/16)))


    print("In A + B': Range of pixel intensity is {} to {}".format(np.min(added), np.max(added)))
    print("In A - B': Range of pixel intensity is {} to {}".format(np.min(subtracted), np.max(subtracted)))

    #print("In Image B': Range of pixel intensity is {} to {}".format(np.min(img_b_prime)/16, np.max(img_b_prime)/16))




    #added = ia.add_imgs(img_a, img_b_prime)
    #subtracted = ia.subtract_imgs(img_a, img_b_prime)


    zeroes = np.zeros((added.shape[0], added.shape[1], 3), 'uint16')



    #cv2.imshow("temp", temp)
    #cv2.waitKey(10000)

    #print(temp)



    added_bgr_array = np.add(zeroes.copy(), sixteen_bit_max).copy()
    added_bgr_array[:, :, 2] = func(added)[2]
    added_bgr_array[:, :, 1] = func(added)[1]
    added_bgr_array[:, :, 0] = func(added)[0]

    subtracted_bgr_array = np.add(zeroes.copy(), sixteen_bit_max).copy()
    subtracted_bgr_array[:, :, 2] = func(subtracted)[2]
    subtracted_bgr_array[:, :, 1] = func(subtracted)[1]
    subtracted_bgr_array[:, :, 0] = func(subtracted)[0]


    #print(bgr_array)
    #print(np.max(temp))

    cv2.imshow("added", added_bgr_array)
    cv2.waitKey(10000)





    cv2.imshow("subtracted", subtracted_bgr_array)
    cv2.waitKey(10000)


    #print(np.min(bgr_array_sub))

    #bgr_array[:, :, 1] = np.where(temp[:, :, 1] < 0,
                               #   0
                                #  ,sixteen_bit_max)

    #bgr_array[:, :, 0] = np.where(temp[:, :, 0] < 0,
                                  #0
                                 #,sixteen_bit_max)

    #bgr_array[:, :, 2] = np.where(temp[:, :, 2] > 2,
                                  #bgr_array[:, :, 2] - (8 * temp[:, :, 2])
                                  #, bgr_array[:, :, 2])

    #bgr_array[:, :, 1] = np.where(temp[:, :, 1] > 1,
                                  #bgr_array[:, :, 1] - (8 * temp[:, :, 1])
                                  #, bgr_array[:, :, 1])



    #cv2.imshow("Subtracted", bgr_array_sub)
    #cv2.waitKey(10000)

    #bgr_array[:, :, 1] = bgr_array[:, :, 1] - (8 * temp[:, :, 1])
    #bgr_array[:, :, 0] = bgr_array[:, :, 0] - (8 * temp[:, :, 0])
    #cac.create_colormap("plt_added.png", test_materials, added)
    #cac.create_colormap("plt_subtracted.png", test_materials, subtracted)


    print("Done")

    """
    print("Test 1a: Coregistration Module\n")

    print("Testing package dependencies:")

    current_package = ""
    try:
        current_package = "cv2"
        import cv2
        print("\t{} import: {}".format(current_package, "Pass"))
        del cv2

        current_package = "numpy"
        import numpy
        print("\t{} import: {}".format(current_package, "Pass"))
        del numpy

    except (ImportError, ModuleNotFoundError) as e:
        raise Exception("\t{} import: {}".format(current_package, "Fail"))

    print("\nStarting Test 1a.")

    coreg_import_check_string = "\tTesting coregistration module import: "


    try:
        import test_coregistration_module
        print(coreg_import_check_string + "Pass")
    except (ImportError, ModuleNotFoundError) as e:
        print(coreg_import_check_string + "Fail")
        print("\tFor more detailed information, see the stack trace below.\n")
        raise e

    directory_check_string = "\tTest materials present: {}"
    test_materials = os.path.join(os.path.join(os.getcwd(), "tests"), "2019_12_14__13_08")
    if os.path.exists(test_materials):
        print(directory_check_string.format("Yes"))
    else:
        raise Exception("Can not run tests. You do not have your tests module set up properly.\n"
                        "Obtain the 2019_12_14__13_08 run and add it to this directory, then try again.")

    cam_a_image_path = os.path.join(os.path.join(test_materials, "cam_a_frames"), "cam_a_frame_1.png")
    cam_b_image_path = os.path.join(os.path.join(test_materials, "cam_b_frames"), "cam_b_frame_1.png")

    if not os.path.exists(cam_a_image_path):
        raise Exception("Can not run this test_RminRmax. You do not have your tests module set up properly.\n"
                        "Obtain image {} and add it to the tests directory, then try again.".format(cam_a_image_path))

    if not os.path.exists(cam_b_image_path):
        raise Exception("Can not run this test_RminRmax. You do not have your tests module set up properly.\n"
                        "Obtain image {} and add it to the tests directory, then try again.".format(cam_b_image_path))

    #print("\n\tStarting test_euclidean_transform()")
    #test_coregistration_module.test_find_ecc()


    print("\n\tStarting test_find_gaussian_profile()")
    test_coregistration_module.test_find_gaussian_profile()

    #print("\n\tStarting test_img_characterization()")
    #test_coregistration_module.test_img_characterization()

    """
