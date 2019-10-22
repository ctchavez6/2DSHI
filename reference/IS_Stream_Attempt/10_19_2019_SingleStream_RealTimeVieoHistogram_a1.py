import os

cwd = os.getcwd()
parent_directory = os.path.dirname(os.getcwd())
configurations_folder = parent_directory + "\ConfigFiles"
configurations_file = configurations_folder + "\/23097552_setup_Oct15.pfs"


from pypylon_opencv_viewer import BaslerOpenCVViewer

# Pypylon get camera by serial number
serial_number = '23097552'

'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

'''
from pypylon import pylon
import cv2

# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
pylon.FeaturePersistence.Load(configurations_file, camera.GetNodeMap()) # Config file with the capturing properties

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    grabResult.Release()

# Releasing the resource
camera.StopGrabbing()

cv2.destroyAllWindows()


""" 



import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from pypylon import pylon
import os

cwd = os.getcwd()
parent_directory = os.path.dirname(os.getcwd())
configurations_folder = parent_directory + "\ConfigFiles"
configurations_file = configurations_folder + "\/23097552_setup_Oct15.pfs"


#y = 0
#a = 0
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', '--file1',
    help='Path to video file (if not using camera)')
parser.add_argument('-c', '--color', type=str, default='gray',
    help='Color space: "gray" (default), "rgb", or "lab"')
parser.add_argument('-b', '--bins', type=int, default=16,
    help='Number of bins per channel (default 16)')
parser.add_argument('-w', '--width', type=int, default=0,
    help='Resize video to specified width in pixels (maintains aspect)')
args = vars(parser.parse_args())


from pypylon_opencv_viewer import BaslerOpenCVViewer

# Pypylon get camera by serial number
serial_number = '23097552'

'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

'''

# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
pylon.FeaturePersistence.Load(configurations_file, camera.GetNodeMap()) # Config file with the capturing properties

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned



# Configure VideoCapture class instance for using camera or file input.
if not args.get('file', False):
    #supply the camera index
    capture = cv2.VideoCapture()
    #capture1 = cv2.VideoCapture(1)
else:
    #supply the filename
    capture = cv2.VideoCapture(args['file'])
    #capture1 = cv2.VideoCapture(args['file1'])

color = args['color']
bins = args['bins']
resizeWidth = args['width']


fig, (ax, ax1) = plt.subplots(1, 2)


ax.set_title('Histogram (grayscale)')
ax1.set_title('Histogram1 (grayscale)')
ax.set_xlabel('Bin')
ax.set_ylabel('Frequency')
ax1.set_xlabel('Bin')
ax1.set_ylabel('Frequency')


# Initialize plot line object(s). Turn on interactive plotting and show plot.
lw = 3
alpha = 0.5

lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw, label='intensity')
lineGray1, = ax1.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw, label='intensity')

# add statistical data to plot
lineMaximum, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='g', lw=1, label='maximum')
#xlineMaximum, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='g', lw=1, label='maximum')
lineAverage, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='b', linestyle='dashed', lw=1, label='average')
lineStdev, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='r', linestyle='dotted',lw=2, label='stdev')
min_xlim, max_xlim = plt.xlim(); min_ylim, max_ylim = plt.ylim()
#plt.text(max_xlim*0.5, max_ylim*0.6, 'Max: {:.2f}'.format(average), color='b')
#textLabel, = ax.text(max_xlim*0.5, max_ylim*0.5, ax.set_xlabel)



ax.set_xlim(0, bins-1)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True)
ax1.set_xlim(0, bins-1)
ax1.set_ylim(0, 1)
ax1.legend()
ax1.grid(True)
#ax.annotate(maximum)
plt.ion()
#plt.text()
plt.show()
#plt1.show()
#cv2.imshow('Frame', ax1)


# ax.annotate(maximum)
# plt1.ion()
# plt.text()
# plt1.show()



while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    grabResult.Release()

    (grabbed, frame) = capture.read()
    #(grabbed1, frame1) = capture1.read()

    if not grabbed:
        print("breaking from:\n\t if not grabbed:\n")
        break
    #if not grabbed1:
     #   print("breaking from:\n\t if not grabbed1:\n")
     #   break

    # Resize frame to width, if specified.
    if resizeWidth > 0:
        (height, width) = frame.shape[:2]
        resizeHeight = int(float(resizeWidth / width) * height)
        frame = cv2.resize(frame, (resizeWidth, resizeHeight),
                           interpolation=cv2.INTER_AREA)

    # Normalize histograms based on number of pixels per frame.
    numPixels = np.prod(frame.shape[:2])

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', gray)
    histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
    lineGray.set_ydata(histogram)

    maximum = np.nanmax(histogram)
    lineMaximum.set_ydata(maximum)
    average = np.average(histogram)
    stdev = np.nanstd(histogram)
    lineAverage.set_ydata(average)
    lineStdev.set_ydata(stdev)

    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', gray1)
    histogram1 = cv2.calcHist([gray1], [0], None, [bins], [0, 255]) / numPixels
    lineGray1.set_ydata(histogram1)

        # ymax_index = np.where(histogram==maximum)
        # xlineMaximum.set_xdata(ymax_index)

        # textLabel.set_ydata(stdev)

    #       dataFile.write(str(b))
    #       e = np.nanmax(np.bincount(histogram))
    #       plt.axhline(b, color='r', linestyle='dashed', linewidth=1)
    #       plt.axhline(c, color='g', linestyle='dashed', linewidth=1)
    #       plt.axhline(d, color='b', linestyle='dashed', linewidth=1)
    #       plt.axhline(e, color='g', linestyle='solid', linewidth=1)

    #       plt.subplot(121), plt.axvline(mean_val.all(), color='k', linestyle='dashed', linewidth=1)
    #       plt.subplot(122), plt.axvline(mean_val, color='k', linestyle='dashed', linewidth=1)

    #    min_ylim, max_ylim = plt.ylim()
    #    plt.text(x.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(x.mean()))
    #    fig.canvas.draw()

    # waits () milliseconds for a keypress, extracts the last 8 bits of the keyed input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # print(type(maximum))
        fig.savefig('rtVideoHisFigure.png')
        print(min_xlim, max_xlim)
        print(min_ylim, max_ylim)
        print(average, stdev)
        print("breaking from:\n\tif cv2.waitKey(1) & 0xFF == ord('q'):\n")
        break
print("Exited While True")
# Releasing the resource
camera.StopGrabbing()
capture.release()
cv2.destroyAllWindows()

"""