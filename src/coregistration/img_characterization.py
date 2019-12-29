import cv2
import numpy as np
import math

def initialize_orb_detector(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20):
    return cv2.ORB_create(nfeatures=nfeatures, scoreType=scoreType, nlevels=nlevels)

def characterize_img(img, orb_detector, mask=None):
    keypoints, descriptors = orb_detector.detectAndCompute(img, mask=mask)
    return keypoints, descriptors

def draw_keypoints(img, keypoints, color=(0, 255, 0)):
    return cv2.drawKeypoints(img, keypoints, color=color, flags=0, outImage=np.array([]))

def find_matches(img1, img2, matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True), threshold=90):
    orb_ = initialize_orb_detector()

    kp1, d1 = characterize_img(img1, orb_)
    kp2, d2 = characterize_img(img2, orb_)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * threshold)]

    return kp1, kp2, matches


def get_homography_components(homography_matrix):
    '''((translationx, translationy), rotation, (scalex, scaley), shear)'''
    a = homography_matrix[0,0]
    b = homography_matrix[0,1]
    c = homography_matrix[0,2]
    d = homography_matrix[1,0]
    e = homography_matrix[1,1]
    f = homography_matrix[1,2]

    p = math.sqrt(a*a + b*b)
    r = (a*e - b*d)/(p)
    q = (a*d+b*e)/(a*e - b*d)

    translation_ = (c,f)
    scale_ = (p,r)
    shear_ = q
    theta_ = math.atan2(b,a)

    return translation_, math.degrees(theta_), scale_, shear_

def derive_homography(img_a_8bit, img_b_8bit, supress_shear=False):

    orb_detector = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20)
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img_a_8bit, None)
    print("A has {} keypoints".format(len(kp1)))

    kp2, d2 = orb_detector.detectAndCompute(img_b_8bit, None)
    print("B has {} keypoints".format(len(kp2)))

    a_with_keypoints = draw_keypoints(img_a_8bit, kp1)
    b_with_keypoints = draw_keypoints(img_b_8bit, kp2)

    cv2.imshow("Cam A: Keypoints", a_with_keypoints)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    cv2.imshow("Cam B: Keypoints", b_with_keypoints)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches_ = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)
    print("matches: ", no_of_matches)
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches_)):
        p1[i, :] = kp1[matches_[i].queryIdx].pt
        p2[i, :] = kp2[matches_[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    homography_components = get_homography_components(homography)
    translation = homography_components[0]
    angle = homography_components[1]
    scale = homography_components[2]
    if supress_shear:
        shear = 0.0

    shear = homography_components[3]

    print("Suggested Angle of Rotation: {}".format(angle))
    print("Suggested translation: {}".format(translation))
    print("Suggested scale: {}".format(scale))
    print("Suggested shear: {}".format(shear))


    return homography


def transform_img(img_b_16bit, homography):
    height, width = img_b_16bit.shape

    transformed_img = cv2.warpPerspective(img_b_16bit, homography, (width, height))
    return transformed_img
