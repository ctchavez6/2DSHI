#from src.path_management import image_management as im
import numpy as np



def add_imgs(img1, img2):
    return np.add(img1, img2)


def subtract_imgs(img1, img2):
    return np.add(img1, img2*(-1))