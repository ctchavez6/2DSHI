# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(args["image"])

image = cv2.blur(image,(50,50))
kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 18,-1],
                              [-1,-1,-1]])
image = cv2.filter2D(image, -1, kernel_sharpening)

# image = cv2.blur(image,(50,50))
# image = image*1.5
# image = image.astype('uint8')
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100,param1= 100,
                           param2= 95, minRadius= 80, maxRadius=900)
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # show the output image
    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)
if circles is None:
    print("none")
    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)