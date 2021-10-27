'''
Copyright (c) 2020 Pantelis Liolios
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from tkinter import Tk, filedialog, messagebox, simpledialog
import tkinter
# import cv
import os

import numpy as np
from numpy import asarray, savetxt

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def run():
    '''
    Main function of curve digitizer
    '''

    # open the dialog box
    # first hide the root window
    root = Tk()
    root.withdraw()
    # open the dialog
    filein = filedialog.askopenfilename(
        title="Select image to digitize",
        filetypes=(
            ("jpeg files", "*.jpg"),
            ("png files", "*.png"))
    )
    if len(filein) == 0:
        # nothing selected, return
        return

    # show the image
    img = mpimg.imread(filein)
    _, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')  # clear x-axis and y-axis
    # ask for the cent point of the stream
    pixelCoord = streamClicker(0)
    # clear the figure
    plt.clf()


def streamClicker(stream):
    # get the center reference point
    reply = False
    while not reply:
        messagebox.showinfo("Select reference point",
                            "Use the mouse to select the reference point. " +
                            "Click the center if the area of interest."
                            )
        # capture only one point
        coord = plt.ginput(
                1,
                timeout=0,
                show_clicks=True)
        coord = np.array(coord)

        coord[0][0] = int(coord[0][0])
        coord[0][1] = int(coord[0][1])
        # ask if the point saved it the correct point
        reply = messagebox.askyesno("Point confirmation",
                                    "You selected pixel {:.0f} {:.0f}. Is this correct?".format(
                                        coord[0][0], coord[0][1])
                                    )
    print(coord)
    print(type(coord),type(coord[0][0]))

    return coord


if __name__ == "__main__":
    '''
    Digitize curves from scanned plots
    '''

    # run the main function
    run()