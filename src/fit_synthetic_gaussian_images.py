import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from scipy import optimize

list_of_images = os.listdir(os.getcwd())
for filename in list_of_images[:]:  # filelist[:] makes a copy of filelist.
    if not (filename.endswith(".png")) or (filename == "center_maybe.png"):
        list_of_images.remove(filename)

def gaus(x, mu, sigma, coefficient=10):
    return coefficient*np.exp(-(x-mu)**2/(2*sigma**2))

def findMiddle(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        return (input_list[int(middle)], input_list[int(middle-1)])

def fit_center_linspace(filename, horizontal=True):
    two_d_image_array = cv2.imread(image_filename, -1)
    height, width = two_d_image_array.shape[0], two_d_image_array.shape[1]
    print("Width:  %s\nHeight: %s" % (str(width), str(height)))
    print("Vertical Center(s): %s" % str(findMiddle(np.arange(0, height))))
    vertical_center = findMiddle(np.arange(0, height))
    if len(findMiddle(np.arange(0, height))) > 1:
        vertical_center = findMiddle(np.arange(0, height))[0]
        print("First Vertical Center: %s" % str(findMiddle(np.arange(0, height))[0]))

    center_row = two_d_image_array[:][vertical_center]
    indices = np.arange(0, len(center_row))
    fig = plt.figure()
    #plt.scatter(indices, center_row, s=1)
    plt.savefig("pattern.png")
    popt, _ = optimize.curve_fit(gaus, indices, center_row)
    plt.plot(indices, gaus(indices, *popt), c='red')

    print(popt)
    #cv2.imwrite("center_maybe.png", center_row)


for image_filename in list_of_images:
    print("\nStarting analysis for: " + image_filename)
    fit_center_linspace(image_filename)
    #data = np.asarray(cv2.imread(image_filename, -1).flatten())
    # data = data[data != 0.0]
    #print(data.shape)
    #print(max(data))
    #print(min(data))
    #print(np.mean(data))
    #plt.hist(data)
    #plt.show()
    print("\n")
    break
