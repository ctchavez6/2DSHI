from pypylon import pylon
import numpy as np
import cv2
from image_processing import bit_depth_conversion as bdc
import os


def keep_streaming(cam_array, one_by_one=False):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    if (not one_by_one) and not cam_array.IsGrabbing():
        return False
    if one_by_one and not cam_array.IsGrabbing():
        return True
    return True

camera_config_files_directory = os.path.join(os.getcwd(), 'camera_configuration_files')
config_file = os.path.join(camera_config_files_directory, 'cam_a_default.pfs')


break_key = 'q'
tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()
if len(devices) == 0:
    raise Exception("No camera present.")

cameras = pylon.InstantCameraArray(2)

for i, camera in enumerate(cameras):
    camera.Attach(tlFactory.CreateDevice(devices[i]))
    camera.Open()
    pylon.FeaturePersistence.Load(config_file, camera.GetNodeMap())

# Starts grabbing for all cameras
cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly,
                      pylon.GrabLoop_ProvidedByUser)

continue_stream = True

while continue_stream:
        grabResult1 = cameras[0].RetrieveResult(5000,
                                                pylon.TimeoutHandling_ThrowException)

        grabResult2 = cameras[1].RetrieveResult(5000,
                                                pylon.TimeoutHandling_ThrowException)

        if grabResult1.GrabSucceeded() & grabResult2.GrabSucceeded():
            im1 = grabResult1.GetArray()
            im2 = grabResult2.GetArray()
            grabResult1.Release()
            grabResult2.Release()
            im1_16bit, im2_16bit = bdc.to_16_bit(im1), bdc.to_16_bit(im2)

            cv2.imshow("A", im1_16bit)
            cv2.imshow("B Prime", im2_16bit)
            continue_stream = keep_streaming(cameras)


cv2.destroyAllWindows()