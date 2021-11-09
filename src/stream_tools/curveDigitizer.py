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

from tkinter import Tk, messagebox
from src.image_processing import bit_depth_conversion as bdc
import tkinter
import time
import cv2

coordsA = []
coordsB = []
frameA = None
frameB = None


def run(stream):
    '''
    Main function of curve digitizer
    '''

    global frameA, frameB, coordsA, coordsB

    # open the dialog box
    # first hide the root window
    root = Tk()
    root.withdraw()

    messagebox.showinfo("Digitize curve",
                        "Please digitize the curve. \nThe first point is the origin. \n" +
                        "Left click: select point; \nRight click: undo; \nq: finish"
                        )
    line = []
    setLoop = True
    reply = True
    yellow = (0, 250 * 256, 250 * 256)
    red = (0, 10 * 256, 255 * 256)
    blue = (255 * 256, 10 * 256, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while setLoop:
        i = 1
        # open the dialog
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames(warp_matrix=stream.warp_matrix)
        stream.current_frame_a = cv2.cvtColor(stream.current_frame_a, cv2.COLOR_GRAY2BGR)
        stream.current_frame_b = cv2.cvtColor(stream.current_frame_b, cv2.COLOR_GRAY2BGR)
        a_as_16bit = bdc.to_16_bit(stream.current_frame_a)
        b_as_16bit = bdc.to_16_bit(stream.current_frame_b)
        frameA = a_as_16bit
        frameB = b_as_16bit

        if len(coordsA) > 0:
            lineFrame = cv2.circle(frameA, (coordsA[0][0], coordsA[0][1]), 4, red, thickness=1, lineType=8, shift=0)
            cv2.putText(lineFrame,
                        str(coordsA[0][0]) + ', ' + str(coordsA[0][1]),
                        (coordsA[0][0] + 10, coordsA[0][1] - 20),
                        font,
                        0.6,
                        blue,
                        1)
        if len(coordsA) > 1:
            if i < len(coordsA):
                window_name = frameA
                start_point1 = (coordsA[i-1][0], coordsA[i-1][1])
                end_point1 = (coordsA[i][0], coordsA[i][1])
                thick = 1
                line.append([start_point1, end_point1])
                while i < len(coordsA):
                    lineFrame = cv2.line(window_name, coordsA[i-1], coordsA[i], red, thick)
                    cv2.imshow('webcamA', lineFrame)
                    cv2.putText(lineFrame,
                                str(coordsA[i][0]) + ', ' + str(coordsA[i][1]),
                                (coordsA[i][0] + 10, coordsA[i][1] - 20),
                                font,
                                0.6,
                                blue,
                                1)
                    cv2.imshow('webcamA', lineFrame)
                    i += 1
        if len(coordsB) > 0:
            lineFrame = cv2.circle(frame, (coordsB[0][0], coordsB[0][1]), 4, red, thickness=1, lineType=8, shift=0)
            cv2.putText(lineFrame,
                        str(coordsB[0][0]) + ', ' + str(coordsB[0][1]),
                        (coordsB[0][0] + 10, coordsB[0][1] - 20),
                        font,
                        0.6,
                        blue,
                        1)
        if len(coordsB) > 1:
            if i < len(coordsB):
                window_name = frameB
                start_point1 = (coordsB[i-1][0], coordsB[i-1][1])
                end_point1 = (coordsB[i][0], coordsB[i][1])
                thick = 1
                line.append([start_point1, end_point1])
                while i < len(coordsB):
                    lineFrame = cv2.line(window_name, coordsB[i-1], coordsB[i], red, thick)
                    cv2.imshow('webcamB', lineFrame)
                    cv2.putText(lineFrame,
                                str(coordsB[i][0]) + ', ' + str(coordsB[i][1]),
                                (coordsB[i][0] + 10, coordsB[i][1] - 20),
                                font,
                                0.6,
                                blue,
                                1)
                    cv2.imshow('webcamB', lineFrame)
                    i += 1
        else:
            reply = False
            cv2.imshow('webcamA', frameA)
            cv2.imshow('webcamB', frameB)
        cv2.setMouseCallback('webcamA', streamClickerA)
        cv2.setMouseCallback('webcamB', streamClickerB)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #  final calculations and saving coordinates
            # DO NOT use filedialog in this function. There is a bug that will SEGFAULT the program
            reply = messagebox.askyesno("Finished?",
                                        "Continue choosing points?")
            if not reply:
                setLoop = False
                frame = None
                time.sleep(2)
    # digitize curves until stopped by the user
    cv2.destroyWindow('webcamA')
    cv2.destroyWindow('webcamB')
    time.sleep(1)
    return coordsA, coordsB


def streamClickerA(event, x, y, flags, param):
    global frameA, coordsA
    yellow = (0, 250 * 256, 250 * 256)
    red = (0, 10 * 256, 255 * 256)
    blue = (255 * 256, 10 * 256, 0)
    # get the center reference point
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Cam A: ', x, ' ', y)
        coordsA.append([x, y])
    if event == cv2.EVENT_RBUTTONDOWN:
        coordsA.pop()
    if event == cv2.EVENT_MOUSEMOVE:
        windSize = cv2.getWindowImageRect('webcamA')
        window_name = frameA
        start_point1 = (0, y)
        end_point1 = (windSize[2], y)
        start_point2 = (x, 0)
        end_point2 = (x, windSize[3])
        thick = 1
        cross1 = cv2.line(window_name, start_point1, end_point1, blue, thick)
        cross2 = cv2.line(window_name, start_point2, end_point2, blue, thick)
        cv2.imshow('webcamA', cross1)
        cv2.imshow('webcamA', cross2)


def streamClickerB(event, x, y, flags, param):
    global frameB, coordsB
    yellow = (0, 250 * 256, 250 * 256)
    red = (0, 10 * 256, 255 * 256)
    blue = (255 * 256, 10 * 256, 0)
    # get the center reference point
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Cam B: ', x, ' ', y)
        coordsB.append([x, y])
    if event == cv2.EVENT_RBUTTONDOWN:
        coordsB.pop()
    if event == cv2.EVENT_MOUSEMOVE:
        windSize = cv2.getWindowImageRect('webcamB')
        window_name = frameB
        start_point1 = (0, y)
        end_point1 = (windSize[2], y)
        start_point2 = (x, 0)
        end_point2 = (x, windSize[3])
        thick = 1
        cross1 = cv2.line(window_name, start_point1, end_point1, blue, thick)
        cross2 = cv2.line(window_name, start_point2, end_point2, blue, thick)
        cv2.imshow('webcamB', cross1)
        cv2.imshow('webcamB', cross2)

