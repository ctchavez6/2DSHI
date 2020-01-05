import cv2
from src.coregistration import find_gaussian_profile as gp
import numpy as np
from src.coregistration import img_characterization
from src.path_management import image_management as im
import os


def test_find_ecc():
    print("Testing try_euclidean_transform(gray1, gray2)")

    test_materials = os.path.join(os.path.join(os.getcwd(), "tests"), "2020_01_04__15_56")
    a_frames_dir = os.path.join(test_materials, "cam_a_frames")
    a1 = cv2.imread(os.path.join(a_frames_dir, "a_1.png"), 0)
    #a1_16bit = cv2.imread(os.path.join(a_frames_dir, "a_1.png"), cv2.IMREAD_ANYDEPTH)
    #a1 = a1_16bit

    b_frames_dir = os.path.join(test_materials, "cam_b_frames")
    b1 = cv2.imread(os.path.join(b_frames_dir, "b_1.png"), 0)
    b1_16bit = cv2.imread(os.path.join(b_frames_dir, "b_1.png"), cv2.IMREAD_ANYDEPTH)
    #b1 = b1_16bit

    a1_matrix = np.asarray(a1)
    print("Max a1: {}".format(np.max(a1_matrix.flatten())))

    b1_matrix = np.asarray(b1)
    print("Max b_prime_matrix: {}".format(np.max(b1_matrix.flatten())))

    b1_shape = b1.shape[1], b1.shape[0]

    cv2.imshow("Image A", a1)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


    cv2.imshow("Image B", b1)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


    warp = img_characterization.try_euclidean_transform(a1, b1)

    print("Warp Matrix Below:\n\n{}\n".format(warp))
    a = warp[0][0]
    b = warp[0][1]
    tx = warp[0][2]
    c = warp[1][0]
    d = warp[1][1]
    ty = warp[1][2]




    print("\tTranslation X:{}".format(tx))
    print("\tTranslation Y:{}\n".format(ty))

    scale_x = np.sign(a)*(np.sqrt(a**2 + b**2))
    scale_y = np.sign(d)*(np.sqrt(c**2 + d**2))

    print("\tScale X:{}".format(scale_x))
    print("\tScale Y:{}\n".format(scale_y))

    phi = np.arctan2(-1.0*b, a)
    print("\tPhi Y (rad):{}".format(phi))
    print("\tPhi Y (deg):{}\n".format(np.degrees(phi)))
    #  | cv2.INTER_LINEAR
    b_prime = cv2.warpAffine(b1, warp, b1_shape, flags=cv2.WARP_INVERSE_MAP)
    b_prime_16bit = cv2.warpAffine(b1_16bit, warp, b1_shape, flags=cv2.WARP_INVERSE_MAP)

    b_prime_matrix = np.asarray(b_prime)
    print("Max b_prime_matrix: {}".format(np.max(b_prime_matrix.flatten())))
    im.save_img("B_Prime.png", test_materials, b_prime_16bit)
    #cv2.imwrite(os.path.join(test_materials, "B_Prime.png"), b_prime)

    cv2.imshow("B Prime?", b_prime)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    GOOD_MATCH_PERCENT = 0.5

    orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20)
    keypoints1, descriptors1 = orb.detectAndCompute(a1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(b_prime, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(a1, keypoints1, b_prime, keypoints2, matches, None)
    im.save_img("A_matched_w_B_Prime.png", test_materials, imMatches)

    #cv2.imwrite(os.path.join(test_materials, "A_matched_w_B_Prime.png"), imMatches)
    #imMatches_resized = cv2.resize(imMatches, (imMatches.shape[1], imMatches.shape[0]), interpolation=cv2.INTER_AREA)

    cv2.imshow("A Matched with B Prime", imMatches)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


    #cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)



def test_find_gaussian_profile():
    test_materials = os.path.join(os.path.join(os.getcwd(), "tests"), "2020_01_01__18_01")
    """
    
    a_frames_dir = os.path.join(test_materials, "cam_a_frames")
    a1 = cv2.imread(os.path.join(a_frames_dir, "a_1.png"), 0)
    center_a = gp.get_coordinates_of_maximum(a1)

    print("\t\tTesting get_coordinates_of_maximum(): Pass")
    print("\t\t\tTest Image shape: {}".format(a1.shape))
    print("\t\t\tCenter at {}".format(center_a))
    a1wc = cv2.circle(a1, center_a, 10, (0, 255, 0), 2)
    cv2.imshow("Img A With Key Points", a1wc)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


    """


    b_frames_dir = os.path.join(test_materials, "cam_b_frames")
    b1 = cv2.imread(os.path.join(b_frames_dir, "b_1.png"), 0)
    center_b = gp.get_coordinates_of_maximum(b1)
    print("Image has Shape: ", b1.shape)
    #max_ = 0
    #h, v = None, None
    #for i in range(b1.shape[0]):
        #for j in range(b1.shape[1]):
            #if b1[i, j] > max_:
                #max_ = b1[i, j]
                #h = i
                #v = j

    #print("Maximum Intensity value of {} Recorded at:\n\t({}, {})".format(max_, h, v))




    print("\t\tTesting get_coordinates_of_maximum(): Pass")
    print("\t\t\tTest Image shape: {}".format(b1.shape))
    print("\t\t\tCenter at {}".format(center_b))
    b1wc = cv2.circle(b1.copy(), center_b, 10, (0, 255, 0), 2)
    cv2.imshow("b1 with center", b1wc)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    print("\t\tTesting plot_horizonal_lineout_intensity():")
    #gp.plot_horizonal_lineout_intensity(b1, center_b)
    print()

    print("\t\tTesting get_gaus_boundaries_x():")
    mu_x, sigma_x, amp_x = gp.get_gaus_boundaries_x(b1, center_b)
    #print("\t\t\tMean (Horizontal): {}".format(mu_x))
    #print("\t\t\tStandard Deviation (Horizontal): {}".format(sigma_x,))
    #print("\t\t\tAmplitude (Horizontal): {}".format(amp_x))

    mu_y, sigma_y, amp_y = gp.get_gaus_boundaries_y(b1, center_b)



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
        img_b_with_keypoints = img_characterization.draw_keypoints(img_b, key_points_b)

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

        """
        import matplotlib.pyplot as plt

        #matches = img_characterization.find_matches(img_a, img_b)
        
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

        cv2.imshow("Img B", cv2.resize(img_b, (960, 600), interpolation=cv2.INTER_AREA))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        cv2.imshow("Img B Prime", cv2.resize(img_b_prime, (960, 600), interpolation=cv2.INTER_AREA))
        cv2.waitKey(7500)
        cv2.destroyAllWindows()


        print("\t\tTesting transform_img(): {}".format("Pass"))
        """

    except NameError as e:
        print(characterize_img_test_str.format("Fail"))
        print("\t\tFor more detailed information, see the stack trace below.\n")
        raise e

