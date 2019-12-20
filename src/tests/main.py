import sys
import os

sys.path.append("..")

if __name__ == "__main__":
    print("Welcome to the 2DSHI Software Package tests.\n"
          "Please make sure you are running this script from your src directory.\n"
          "Test 1a: Coregistration Module\n")

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
        #raise Exception()

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
        raise Exception("Can not run this test. You do not have your tests module set up properly.\n"
                        "Obtain image {} and add it to the tests directory, then try again.".format(cam_a_image_path))

    if not os.path.exists(cam_b_image_path):
        raise Exception("Can not run this test. You do not have your tests module set up properly.\n"
                        "Obtain image {} and add it to the tests directory, then try again.".format(cam_b_image_path))

    print("\n\tStarting test_img_characterization()")
    test_coregistration_module.test_img_characterization()

