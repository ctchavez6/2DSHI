import matplotlib.pylab as plt
import numpy as np

class Histocam():
    """This class is the histogram that goes with the camera streams or Image Algebra streams"""
    def __init__(self):
        self.bins = 4095
        self.threshold = 1.2
        self.figure = plt.figure(figsize=(5, 5))
        self.subplot = self.figure.add_subplot()
        self.stream_name = None
        self.histogram_title = "Default Title"

        # camera_identifier = chr(97 + i)

        self.intensity, = self.figure.plot(np.arange(4095), np.zeros((4095, 1)), c='k', lw=3, label='intensity')
        self.maximum, = self.figure.plot(np.arange(4095), np.zeros((4095, 1)), c='k', lw=3, label='intensity')

        self.average, = self.figure.plot(np.arange(4095), np.zeros((4095, 1)), c='k', lw=3, label='intensity')
        self.st_dev, = self.figure.plot(np.arange(4095), np.zeros((4095, 1)), c='k', lw=3, label='intensity')
        self.grayscale_avg, = self.figure.plot(np.arange(4095), np.zeros((4095, 1)), c='k', lw=3, label='intensity')
        self.grayscale_avg_plus, = self.figure.plot(np.arange(4095), np.zeros((4095, 1)), c='k', lw=3, label='intensity')
        self.grayscale_avg_minus, = self.figure.plot(np.arange(4095), np.zeros((4095, 1)), c='k', lw=3, label='intensity')

        # Plan for tomorrow: Finish modularizing Histocam
        lines["maxima"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='g', lw=1, label='maximum')

        lines["averages"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='b', linestyle='dashed', lw=1, label='average')

        lines["stdevs"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='r', linestyle='dotted', lw=2, label='stdev')

        lines["grayscale_avg"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='b', linestyle='dashed', linewidth=1)

        lines["grayscale_avg+0.5sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='r', linestyle='dotted', linewidth=1)

        lines["grayscale_avg-0.5sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='r', linestyle='dotted', linewidth=1)

        lines["max_vert"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='b', linestyle='solid', linewidth=1)

        lines["avg+sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvspan(-100, -100, alpha=0.5, color='#f5beba')

        lines["avg-sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvspan(-100, -100, alpha=0.5, color='#f5beba')




    def update_lines(self,  raw_2d_array):
        pass