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

def find_matches(img1, img2, matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)):
    orb_ = initialize_orb_detector()

    kp1, d1 = characterize_img(img1, orb_)
    kp2, d2 = characterize_img(img2, orb_)

    return matcher.match(d1, d2)  # Match the two sets of descriptors.


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

def derive_homography(matches, kp1, kp2, threshold=90):
    matches.sort(key=lambda x: x.distance)


    matches = matches[:int(len(matches) * threshold)]  # Take the top 90 % matches forward.
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    return homography


def transform_img(original_img, homography):

    return cv2.warpPerspective(original_img, homography, (original_img.shape[0], original_img.shape[1]))
