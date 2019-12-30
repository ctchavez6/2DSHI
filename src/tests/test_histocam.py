import os
from src.histocam import histocam

def test_all():
    test_materials = os.path.join(os.path.join(os.getcwd(), "tests"), "saved_img_stream")
    cam_a_frames_dir = os.path.join(test_materials, "cam_a_frames")
    cam_b_frames_dir = os.path.join(test_materials, "cam_b_frames")

    cam_a_frames_paths = list()
    cam_b_frames_paths = list()

    for subdir, dirs, files in os.walk(cam_a_frames_dir):
        for file in files:
            if file.endswith(".png"):
                cam_a_frames_paths.append(os.path.join(subdir, file))

    for subdir, dirs, files in os.walk(cam_b_frames_dir):
        for file in files:
            if file.endswith(".png"):
                cam_b_frames_paths.append(os.path.join(subdir, file))

    print(cam_a_frames_paths)
    print(cam_b_frames_paths)

