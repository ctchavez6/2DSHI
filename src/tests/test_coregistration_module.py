import cv2
from src.coregistration import img_characterization
import os



#current_submodule = "img_characterization"
#print("Testing {}".format(current_submodule))

def test_img_characterization():
    test_materials = os.path.join(os.path.join(os.getcwd(), "tests"), "2019_12_14__13_08")
    orb_detector_check_string = "\t\tTesting initialize_orb_detector: {}"
    orb_detector = None
    try:
        orb_detector = img_characterization.initialize_orb_detector()
        print(orb_detector_check_string.format("Pass"))
    except NameError as e:
        print(orb_detector_check_string.format("Fail"))
        print("\t\tFor more detailed information, see the stack trace below.\n")
        raise e


    characterize_img_test_str = "\t\tTesting characterize_img(): {}"
    try:
        cam_a_image_path = os.path.join(os.path.join(test_materials, "cam_a_frames"), "cam_a_frame_1.png")
        img_a = cv2.imread(cam_a_image_path, 0)
        key_points, descriptors = img_characterization.characterize_img(img_a, orb_detector)
        print(characterize_img_test_str.format("Pass"))
    except NameError as e:
        print(characterize_img_test_str.format("Fail"))
        print("\t\tFor more detailed information, see the stack trace below.\n")
        raise e

