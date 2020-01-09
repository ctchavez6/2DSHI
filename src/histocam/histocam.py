import matplotlib.pylab as plt
from matplotlib.pyplot import draw
import numpy as np
import cv2


class Histocam():
    """This class is the histogram that goes with the camera streams or Image Algebra streams"""
    def __init__(self):
        self.num_bins = 4096
        self.bins = np.arange(self.num_bins)
        self.zeroes = np.zeros((self.num_bins, 1))
        self.threshold = 1.2
        self.figure = plt.figure(figsize=(5, 5))
        self.subplot = self.figure.add_subplot()
        self.stream_name = None
        self.intensity, = self.subplot.plot(self.bins, self.zeroes, c='k', lw=3, label='intensity')
        self.maximum, = self.subplot.plot(self.bins, self.zeroes, c='g', label='maximum')
        self.average, = self.subplot.plot(self.bins, self.zeroes, c='b', linestyle='dashed', label='average')
        self.st_dev, = self.subplot.plot(self.bins, self.zeroes, c='r', linestyle='dotted', lw=2, label='stdev')
        self.grayscale_avg = self.subplot.axvline(-100, color='b', linestyle='dashed')
        self.grayscale_avg_plus = self.subplot.axvline(-100, color='r', linestyle='dotted')
        self.grayscale_avg_minus = self.subplot.axvline(-100, color='r', linestyle='dotted')
        self.max_vert = self.subplot.axvline(-100, color='g', linestyle='solid', linewidth=1)
        self.avg_plus_sigma = self.subplot.axvspan(-100, -100, alpha=0.5, color='#f5beba')
        self.avg_minus_sigma = self.subplot.axvspan(-100, -100, alpha=0.5, color='#f5beba')

        self.subplot.set_xlim(-100, self.num_bins - 1 + 100)
        self.subplot.grid(True)
        self.subplot.set_autoscale_on(False)
        self.subplot.set_ylim(bottom=0, top=1)


    def set_xvalues(self, polygon, x0, x1):
        """
        Given a rectangular matplotlib.patches.Polygon object sets the horizontal values.

        Args:
            polygon: An instance of tlFactory.EnumerateDevices()
            x0: An integer
            x1: An integer
        Raises:
            Exception: TODO Add some error handling.

        """
        _ndarray = polygon.get_xy()
        len_ndarray = len(_ndarray)

        if len_ndarray == 4:
            _ndarray[:, 0] = [x0, x0, x1, x1]
        if len_ndarray == 5:
            _ndarray[:, 0] = [x0, x0, x1, x1, x0]

        polygon.set_xy(_ndarray)

    def update(self,  raw_2d_array):
        calculated_hist = cv2.calcHist([raw_2d_array], [0], None, [self.num_bins], [0, self.num_bins-1]) / np.prod(raw_2d_array.shape[:2])

        histogram_maximum = np.amax(calculated_hist)
        greyscale_max = np.amax(raw_2d_array.flatten())
        greyscale_avg = np.mean(raw_2d_array)
        greyscale_stdev = np.std(raw_2d_array)

        self.intensity.set_ydata(calculated_hist)  # Intensities/Percent of Saturation
        self.maximum.set_ydata(greyscale_max)  # Maximums
        self.average.set_ydata(greyscale_avg)  # Averages
        self.st_dev.set_ydata(greyscale_stdev)  # Standard Deviations
        self.grayscale_avg.set_xdata(greyscale_max)  # Maximum Indicator as vertical line
        self.grayscale_avg_plus.set_xdata(min([self.num_bins, greyscale_avg + (greyscale_stdev*0.5)]))  # Vert Line
        self.grayscale_avg_minus.set_xdata(max([self.num_bins, greyscale_avg - (greyscale_stdev * 0.5)]))  # Vert Line

        self.set_xvalues(self.avg_minus_sigma, greyscale_avg, min([self.num_bins, greyscale_avg + (greyscale_stdev * 0.5)]))
        self.set_xvalues(self.avg_minus_sigma, max([greyscale_avg - (greyscale_stdev * 0.5), 0]),greyscale_avg)

        self.subplot.legend(
            labels=(
                "intensity",
                "maximum %.0f" % greyscale_max,
                "average %.2f" % greyscale_avg,
                "stdev %.4f" % greyscale_stdev,),
            loc="upper right"
        )
        if histogram_maximum > 0.001:
            self.subplot.set_ylim(bottom=0.000000, top=histogram_maximum * self.threshold)
        else:
            self.subplot.set_ylim(bottom=0.000000, top=0.001)
        self.figure.canvas.draw()


    def get_plot(self):
        plot = np.fromstring(self.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        plot = plot.reshape(self.figure.canvas.get_width_height()[::-1] + (3,))
        plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr

        return plot

    def get_figure(self):
        return self.figure
