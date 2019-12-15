import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
from PIL import Image

run = "2019_12_14__13_08"



def save_img(filename, directory, image, sixteen_bit=True):
  """
  TODO Add documentation.
  """
  os.chdir(directory)
  if sixteen_bit:
    image = Image.fromarray(image)
    image.save(filename, compress_level=0)
  else:
    cv2.imwrite(filename, image.astype(np.uint16))
  os.chdir(directory)


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

if not os.path.exists(os.path.join(img_a_path, "Image_Algebra")):
  os.mkdir(os.path.join(img_a_path, "Image_Algebra"))

algebra_directory = os.path.join(img_a_path, "Image_Algebra")


img_a_path = os.path.join(img_a_path, "cam_a_frames")
img_a_path = os.path.join(img_a_path, "cam_a_frame_1.png")

img_a = cv2.imread(img_a_path, 0)
# cv2.IMREAD_ANYDEPTH

img_b_path = os.path.join("D:", "2DSHI_Runs")
img_b_path = os.path.join(img_b_path, run)
img_b_path = os.path.join(img_b_path, "cam_b_frames")
img_b_path = os.path.join(img_b_path, "cam_b_frame_1.png")

img_b = cv2.imread(img_b_path, 0)

img_a_8bit = img_a.astype('uint8')
img_b_8bit = img_b.astype('uint8')


print("Img A Shape:", img_a.shape)

print("Img B Shape:", img_b.shape)



#rotated_img = cv2.imread(img_a_path, 0)
#rotated_img = np.rot90(rotated_img, 1)

height, width = img_a.shape
img_a_color = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
img_b_color = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
#blank_image = np.zeros((height,width,3), np.uint8)

# Create ORB detector with 5000 features.
#orb_detector = cv2.ORB_create(5000)
orb_detector = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20)
# Find keypoints and descriptors.
# The first arg is the image, second arg is the mask
#  (which is not required in this case).
kp1, d1 = orb_detector.detectAndCompute(img_a_8bit, None)
kp2, d2 = orb_detector.detectAndCompute(img_b_8bit, None)



print("n-keypoints img 1: ", len(kp1))
print("n-keypoints img 2: ", len(kp2))
#for marker in kp1:
#  img_a_w_keypoints = cv2.drawMarker(img_a_color, tuple(int(i) for i in marker.pt), color=(0, 255, 0))

#for marker in kp2:
#  img_b_w_keypoints = cv2.drawMarker(img_b_color, tuple(int(i) for i in marker.pt), color=(0, 255, 0))


img_a_w_keypoints = cv2.drawKeypoints(img_a, kp1, color=(0, 255, 0), flags=0, outImage=np.array([]))
img_b_w_keypoints = cv2.drawKeypoints(img_b, kp2, color=(0, 255, 0), flags=0, outImage=np.array([]))

f = plt.figure()
original = f.add_subplot(1, 2, 1)
original.title.set_text('Img A')
plt.imshow(img_a_w_keypoints, cmap='gray')
rotated = f.add_subplot(1,2, 2)
rotated.title.set_text('Img B')
plt.imshow(img_b_w_keypoints, cmap='gray')
plt.show(block=True)
plt.close('all')


# Match features between the two images.
# We create a Brute Force matcher with
# Hamming distance as measurement mode.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Match the two sets of descriptors.
matches = matcher.match(d1, d2)

# Sort matches on the basis of their Hamming distance.
matches.sort(key=lambda x: x.distance)

# Take the top 90 % matches forward.
matches = matches[:int(len(matches)*90)]
no_of_matches = len(matches)
print("matches: ", no_of_matches)

plt.close('all')


fig = plt.figure()
img3 = cv2.drawMatches(img_a,kp1,img_b,kp2,matches[:100],None, flags=2)
plt.imshow(img3)
plt.title("Img_7_2019_12_11__20_32")
fig.savefig("Img_7_2019_12_11__20_32.png")
plt.show()

plt.close('all')

# Define empty matrices of shape no_of_matches * 2.
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))

for i in range(len(matches)):
  p1[i, :] = kp1[matches[i].queryIdx].pt
  p2[i, :] = kp2[matches[i].trainIdx].pt

# Find the homography matrix.
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)


# Use this matrix to transform the
# colored image wrt the reference image.
#transformed_img = cv2.warpPerspective(img,
#                    homography, (width, height))

# Save the output.
#cv2.imwrite('output.png', transformed_img)

homography_components = getComponents(homography)
translation = homography_components[0]
angle = homography_components[1]
scale = homography_components[2]
shear = homography_components[3]

print("Suggested Angle of Rotation: {}".format(angle))

img_b_16bit = cv2.imread(img_a_path, cv2.IMREAD_ANYDEPTH)

transformed_img = cv2.warpPerspective(img_b_16bit,
                    homography, (width, height))

print("Max of img A:", np.max(img_a.flatten()))
print("Max of img B:", np.max(img_b.flatten()))
print("Max of transformed img B:", np.max(transformed_img.flatten()))
cv2.imshow("B-Prime", transformed_img)
cv2.waitKey(10000)

save_img(os.path.join(algebra_directory, "B_Prime" + ".png"), os.getcwd(), transformed_img)