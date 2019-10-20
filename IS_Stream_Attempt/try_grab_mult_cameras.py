"""How to grab and process images from multiple cameras using the CInstantCameraArray class. The CInstantCameraArray
class represents an array of instant camera objects. It provides almost the same interface as the instant camera for
grabbing. The main purpose of the CInstantCameraArray is to simplify waiting for images and camera events of multiple
cameras in one thread. This is done by providing a single RetrieveResult method for all cameras in the array.
Alternatively, the grabbing can be started using the internal grab loop threads of all cameras in the
CInstantCameraArray. The grabbed images can then be processed by one or more  image event handlers. """

import os

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys
import cv2

cwd = os.getcwd()
parent_directory = os.path.dirname(os.getcwd())
configurations_folder = parent_directory + "\ConfigFiles"
configurations_file_a = configurations_folder + "\/23170624_setup_Oct15.pfs"
configurations_file_b = configurations_folder + "\/23170624_setup_Oct15.pfs"

"""
Limits the amount of cameras used for grabbing. It is important to manage the available bandwidth when grabbing with 
multiple cameras. This applies, for instance, if two GigE cameras are connected to the same network adapter via a 
switch. To manage the bandwidth, the GevSCPD interpacket delay parameter and the GevSCFTD transmission delay parameter 
can be set for each GigE camera device. The "Controlling Packet Transmission Timing with the Interpacket and Frame 
Transmission Delays on Basler GigE Vision Cameras" Application Notes (AW000649xx000) provide more information about 
this topic. The bandwidth used by a FireWire camera device can be limited by adjusting the packet size.
"""
maxCamerasToUse = 2 # Number of cameras we plan on using
exitCode = 0 # The exit code of the sample application.

try:
    tlFactory = pylon.TlFactory.GetInstance() # Get the transport layer factory.
    devices = tlFactory.EnumerateDevices() # Get all attached devices and exit application if no device is found.

    if len(devices) == 0:
        raise pylon.RUNTIME_EXCEPTION("No cameras present.")

    # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

    l = cameras.GetSize()

    # Create and attach all Pylon Devices.
    cam_a = None
    cam_b = None

    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        print("Camera ", i, "- Using device ", cam.GetDeviceInfo().GetModelName())  # Print camera model number
        cam.Open()

        if i == 0:
            pylon.FeaturePersistence.Load(configurations_file_a, cam.GetNodeMap())
            cam_a = cam
        if i == 1:
            pylon.FeaturePersistence.Load(configurations_file_b, cam.GetNodeMap())
            cam_b = cam

    """Starts grabbing for all cameras starting with index 0. The grabbing is started for one camera after the other.
    That's why the images of all cameras are not taken at the same time. However, a hardware trigger setup can be used
    to cause all cameras to grab images synchronously. According to their default configuration, the cameras are set up 
    for free-running continuous acquisition."""
    cameras.StartGrabbing()
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while cameras.IsGrabbing():
        grabResult_a = cam_a.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grabResult_b = cam_b.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        # print(grabResult)
        if grabResult_a.GrabSucceeded() and grabResult_b.GrabSucceeded():
            # Access the image data
            image_a = converter.Convert(grabResult_a)
            img_a = image_a.GetArray()

            image_b = converter.Convert(grabResult_b)
            img_b = image_b.GetArray()

            cv2.namedWindow('title_a', cv2.WINDOW_NORMAL)
            cv2.imshow('title_a', img_a)

            cv2.namedWindow('title_b', cv2.WINDOW_NORMAL)
            cv2.imshow('title_b', img_b)

            k = cv2.waitKey(1)
            if k == 27:
                break
        grabResult_a.Release()
        grabResult_b.Release()


except genicam.GenericException as e:
    # Error handling
    print("An exception occurred.", e.GetDescription())
    exitCode = 1

# Comment the following two lines to disable waiting on exit.
sys.exit(exitCode)