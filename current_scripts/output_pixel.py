import cv2
import os.path
#image = cv2.imread('images/fw.png', cv2.IMREAD_COLOR)
print(os.getcwd())

parent_directory = os.path.dirname(os.getcwd())  # String representing parent directory of current working directory
print(parent_directory)
print(os.path.exists(parent_directory))
image_path = parent_directory + '\\reference\\images\\fw.jpg'
print(os.path.exists(image_path))
image = cv2.imread(image_path)



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

shape = gray.shape
print(shape)
height = shape[0]
width = shape[1]

f = open("widthHeight.txt", "a")
f.write("height =")
f.write('{:03d}\n'.format(height))
f.write("width =")
f.write('{:03d}\n'.format(width))
f.close

for row in range(width):
    for column in range(height):
        print(gray[column][row])