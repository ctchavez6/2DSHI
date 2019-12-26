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
        key_points_a, descriptors_a = img_characterization.characterize_img(img_a, orb_detector)
        img_a_with_keypoints = img_characterization.draw_keypoints(img_a, key_points_a)

        cam_b_image_path = os.path.join(os.path.join(test_materials, "cam_b_frames"), "cam_b_frame_1.png")
        img_b = cv2.imread(cam_b_image_path, 0)
        key_points_b, descriptors_b = img_characterization.characterize_img(img_b, orb_detector)
        img_b_with_keypoints = img_characterization.draw_keypoints(img_a, key_points_b)

        cv2.imshow("Img A", cv2.resize(img_a, (960, 600), interpolation=cv2.INTER_AREA))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        cv2.imshow("Img A With Key Points", cv2.resize(img_a_with_keypoints, (960, 600), interpolation=cv2.INTER_AREA))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

        cv2.imshow("Img B", cv2.resize(img_b, (960, 600), interpolation=cv2.INTER_AREA))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        cv2.imshow("Img B With Key Points", cv2.resize(img_b_with_keypoints, (960, 600), interpolation=cv2.INTER_AREA))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

        print("\t\tTesting draw_keypoints(): {}".format("Pass"))  # Sort matches on the basis of their Hamming distance.

        import matplotlib.pyplot as plt

        matches = img_characterization.find_matches(img_a, img_b)

        img3 = cv2.drawMatches(img_a, key_points_a, img_b, key_points_b, matches[:100], None, flags=2)
        cv2.imshow("Matches", img3)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

        img_b = cv2.imread(cam_b_image_path, cv2.IMREAD_ANYDEPTH)
        homography = img_characterization.derive_homography(matches, key_points_a, key_points_b)
        img_b_prime = img_characterization.transform_img(img_b, homography)


        translation, angle, scale, shear = img_characterization.get_homography_components(homography)
        components = {"translation": translation, "angle": angle, "scale": scale, "shear": shear}

        for key in components:
            if key == "translation":
                print("\t\t\t{}_x: {}".format(key, round(float(components[key][0]), 2)))
                print("\t\t\t{}_y: {}".format(key, round(float(components[key][1]), 2)))
            else:
                print("\t\t\t{}: {}".format(key, components[key]))
                #print("\t\t\t{}: {}".format(key, round(float(components[key]), 2)))

        cv2.imshow("Img B", cv2.resize(img_b, (960, 600), interpolation=cv2.INTER_AREA))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        cv2.imshow("Img B Prime", cv2.resize(img_b_prime, (960, 600), interpolation=cv2.INTER_AREA))
        cv2.waitKey(7500)
        cv2.destroyAllWindows()

        print("\t\tTesting transform_img(): {}".format("Pass"))
    except NameError as e:
        print(characterize_img_test_str.format("Fail"))
        print("\t\tFor more detailed information, see the stack trace below.\n")
        raise e

