import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import numpy as np
import cv2
import math
import os
from PIL import Image




run = "2019_12_14__13_08"

algebra_directory = os.path.join("D:", "2DSHI_Runs")
algebra_directory = os.path.join(algebra_directory, run)
algebra_directory = os.path.join(algebra_directory, "Image_Algebra")

subtracted_img_path = os.path.join(algebra_directory, "A_Minus_B_Prime.png")
added_img_path = os.path.join(algebra_directory, "A_Plus_B_Prime.png")

a_minus_b_prime = cv2.imread(subtracted_img_path, cv2.IMREAD_ANYDEPTH)
a_plus_b_prime = cv2.imread(added_img_path, cv2.IMREAD_ANYDEPTH)

cv2.imshow("minus", a_minus_b_prime)
cv2.waitKey(5000)
# im.save('test_hot.jpg')
cv2.imshow("plus", a_plus_b_prime)
cv2.waitKey(5000)
# im.save('test_hot.jpg')


# https://stackoverflow.com/questions/43457308/is-there-any-good-color-map-to-convert-gray-scale-image-to-colorful-ones-using-p



#t = timeit.timeit(pil_test, number=1)
def create_colormap(file_name, save_directory, img, original_bit_depth=12, input_bit_depth=16):
    save_path = os.path.join(save_directory, file_name)
    fig = plt.figure()
    intended_img = img/(2**(input_bit_depth-original_bit_depth))
    print(np.max(intended_img.flatten()))
    imgplot = plt.imshow(intended_img)
    imgplot.set_cmap('nipy_spectral')
    plt.colorbar()
    plt.show()
    fig.savefig(save_path)
#print('PIL: %s' % t)
#t = timeit.timeit(plt_test, number=1)
#print('PLT: %s' % t)


create_colormap("A_Minus_B_Prime_Colormap.png", algebra_directory, a_minus_b_prime)
create_colormap("A_Plus_B_Prime_Colormap.png", algebra_directory, a_plus_b_prime)
