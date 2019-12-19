import cv2
import numpy as np

def initialize_orb_detector(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20):
    return cv2.ORB_create(nfeatures=nfeatures, scoreType=scoreType, nlevels=nlevels)


def characterize_img(img, orb_detector, mask=None):
    keypoints, descriptors = orb_detector.detectAndCompute(img, mask=mask)
    return keypoints, descriptors


def draw_keypoints(img, keypoints, color=(0, 255, 0)):
    return cv2.drawKeypoints(img, keypoints, color=color, flags=0, outImage=np.array([]))
