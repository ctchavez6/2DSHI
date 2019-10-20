import IS_Stream_Attempt.stream_a52 as stream_a52 # Un comment when running from Pycharm
#import stream_a52 as stream_a52 # Un comment when running from command line
import IS_Stream_Attempt.stream_b24 as stream_b24 # Un comment when running from Pycharm
#import stream_b24 as stream_b24 # Un comment when running from command line

stream_cam_a52 = True
stream_cam_b24 = False

opened_camera_a = None
opened_camera_b = None

if stream_cam_a52:
    opened_camera_a = stream_a52.open_camera()
    stream_a52.go(opened_camera_a)

if stream_cam_b24:
    opened_camera_b = stream_b24.open_camera()
    stream_b24.go(opened_camera_b)



try:
    while True:
        pass
except KeyboardInterrupt:
    stream_a52.stop(opened_camera_a)
    stream_b24.stop(opened_camera_a)

#if stream_cam_b24:
    #stream_cam_b24.go()

#while stream_cam_a52 or stream_cam_b24:
    #if not stream_cam_a52:



#import IS_Stream_Attempt.stream_b24 as stream_b24
"""

keep_streaming = True  




    """