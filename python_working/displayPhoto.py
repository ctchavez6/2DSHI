import cv2

print("OpenCV version:", cv2.__version__)

img = cv2.imread("fw.png")
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("dragon", img)
#cv2.imshow("dragon - gray", gray)

#type "escape key" anywhere in the photo window
cv2.waitKey(0)
cv2.destroyAllWindows()
