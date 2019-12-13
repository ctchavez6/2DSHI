import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os

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

  translation = (c,f)
  scale = (p,r)
  shear = q
  theta = math.atan2(b,a)

  return (translation, math.degrees(theta), scale, shear)


run = "2019_12_12__18_41"
end_result = os.path.join("D:", "Img_Keypoint_Analysis")
end_result = os.path.join(end_result, "Img_{}_{}".format("N", run))



img_a_path = os.path.join("D:", "2DSHI_Runs")
img_a_path = os.path.join(img_a_path, run)
img_a_path = os.path.join(img_a_path, "cam_a_frames")
img_a_path = os.path.join(img_a_path, "cam_a_frame_1.png")

img_a = cv2.imread(img_a_path, 0)


img_b_path = os.path.join("D:", "2DSHI_Runs")
img_b_path = os.path.join(img_b_path, run)
img_b_path = os.path.join(img_b_path, "cam_b_frames")
img_b_path = os.path.join(img_b_path, "cam_b_frame_1.png")

img_b = cv2.imread(img_b_path, 0)




#rotated_img = cv2.imread(img_a_path, 0)
#rotated_img = np.rot90(rotated_img, 1)

height, width = img_a.shape


# Create ORB detector with 5000 features.
#orb_detector = cv2.ORB_create(5000)
orb_detector = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20)
# Find keypoints and descriptors.
# The first arg is the image, second arg is the mask
#  (which is not reqiured in this case).
kp1, d1 = orb_detector.detectAndCompute(img_a, None)
kp2, d2 = orb_detector.detectAndCompute(img_b, None)



img_a_w_keypoints = cv2.drawKeypoints(img_a, kp1, color=(0, 255, 0), flags=0, outImage=np.array([]))
img_b_w_keypoints = cv2.drawKeypoints(img_b, kp2, color=(0, 255, 0), flags=0, outImage=np.array([]))

f = plt.figure()
original = f.add_subplot(1,2, 1)
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

plt.close('all')

print("n-keypoints img 1: ", len(kp1))
print("n-keypoints img 2: ", len(kp2))

print("matches: ", len(matches))

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


print(getComponents(homography))



