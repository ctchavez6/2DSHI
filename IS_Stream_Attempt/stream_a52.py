import os
from pypylon import pylon
import cv2

def open_camera():
    cwd = os.getcwd()
    parent_directory = os.path.dirname(os.getcwd())
    configurations_folder = parent_directory + "\ConfigFiles"
    configurations_file = configurations_folder + "\/23097552_setup_Oct15.pfs"
    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    pylon.FeaturePersistence.Load(configurations_file, camera.GetNodeMap()) # Config file with the capturing properties
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return camera

def go(opened_camera):

    '''
    A simple Program for grabing video from basler camera and converting it to opencv img.
    Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)
    
    '''
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while opened_camera.IsGrabbing():
        grabResult = opened_camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
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

def stop(opened_camera):
    # Releasing the resource
    opened_camera.StopGrabbing()
    cv2.destroyAllWindows()


cam = open_camera()
go(cam)
