import numpy as np
import cv2
import math
import os
from PIL import Image

#TODO: Research if we are using add_images.py anywhere, if so delete bad code and duplicate functions, else delete.
run = "2019_12_14__13_08"


def save_img(filename, directory, image, sixteen_bit=True):
  """
  TODO Add documentation.
  """
  home_directory = os.getcwd()
  os.chdir(directory)
  if sixteen_bit:
    image = Image.fromarray(image)
    image.save(filename, compress_level=0)
  else:
    cv2.imwrite(filename, image.astype(np.uint16))
  os.chdir(home_directory)


def getComponents(normalised_homography):
  '''((translationx, translationy), rotation, (scalex, scaley), shear)'''
  a = normalised_homography[0,0]
  b = normalised_homography[0,1]
  c = normalised_homography[0,2]
  d = normalised_homography[1,0]
  e = normalised_homography[1,1]
  f = normalised_homography[1,2]

  p = math.sqrt(a*a + b*b)
  r = (a*e - b*d)/(p)
  q = (a*d+b*e)/(a*e - b*d)

  translation_ = (c,f)
  scale_ = (p,r)
  shear_ = q
  theta_ = math.atan2(b,a)

  return (translation_, math.degrees(theta_), scale_, shear_)


def getComponents_mod(normalised_homography):
  '''((translationx, translationy), rotation, (scalex, scaley), shear)'''
  a = normalised_homography[0,0]
  b = normalised_homography[0,1]
  c = normalised_homography[0,2]
  d = normalised_homography[1,0]
  e = normalised_homography[1,1]
  f = normalised_homography[1,2]

  p = math.sqrt(a*a + b*b)
  r = (a*e - b*d)/(p)
  q = (a*d+b*e)/(a*e - b*d)

  translation_ = (c,f)
  scale_ = (p,r)
  shear_ = q
  theta_ = math.atan2(b,a)

  return (translation_, math.degrees(theta_), 0.00, 0.00)





#end_result = os.path.join(end_result, "Img_{}_{}".format("N", run))



img_a_path = os.path.join("D:", "2DSHI_Runs")
img_a_path = os.path.join(img_a_path, run)
algebra_directory = os.path.join(img_a_path, "Image_Algebra")


if not os.path.exists(os.path.join(img_a_path, "Image_Algebra")):
  os.mkdir(algebra_directory)

algebra_directory = os.path.join(img_a_path, "Image_Algebra")


img_a_path = os.path.join(img_a_path, "cam_a_frames")
img_a_path = os.path.join(img_a_path, "cam_a_frame_1.png")

img_a = cv2.imread(img_a_path, cv2.IMREAD_ANYDEPTH)
# cv2.IMREAD_ANYDEPTH

img_b_path = os.path.join("D:", "2DSHI_Runs")
img_b_path = os.path.join(img_b_path, run)
img_b_path = os.path.join(img_b_path, "Image_Algebra")
img_b_path = os.path.join(img_b_path, "B_Prime.png")


img_b_prime = cv2.imread(img_a_path, cv2.IMREAD_ANYDEPTH)







a_plus_b_prime = cv2.add(img_b_prime, img_a)

cv2.imshow("A", img_a)
cv2.waitKey(5000)

cv2.imshow("B_Prime", img_b_prime)
cv2.waitKey(5000)

cv2.imshow("A Plus B Prime", a_plus_b_prime)
cv2.waitKey(5000)

save_img("A_Plus_B_Prime.png", algebra_directory, a_plus_b_prime)


