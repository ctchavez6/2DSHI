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
import tkinter
import time
import cv2
import archiveHelper as ah

frame = None
coords = []


def run():
    '''
    Main function of curve digitizer
    '''

    global frame, coords

    # open the dialog box
    # first hide the root window
    root = Tk()
    root.withdraw()
    # open the dialog
    cam = cv2.VideoCapture(0)  # camera index (default = 0) (added based on Randyr's comment).

    print('cam has image : %s' % cam.read()[0])  # True = got image captured.
    # False = no pics for you to shoot at.

    # Lets check start/open your cam!
    if cam.read() == False:
        cam.open()

    if not cam.isOpened():
        print('Cannot open camera')
    else:
        messagebox.showinfo("Digitize curve",
                            "Please digitize the curve. \nThe first point is the origin. \n" +
                            "Left click: select point; \nRight click: undo; \nq: finish"
                            )
        line = []
        setLoop = True
        reply = True
        while setLoop:
            i = 1
            ret, frame = cam.read()
            if len(coords) > 1:
                if i < len(coords):
                    window_name = frame
                    start_point1 = (coords[i-1][0], coords[i-1][1])
                    end_point1 = (coords[i][0], coords[i][1])
                    color = (0, 250, 250)
                    thick = 1
                    line.append([start_point1, end_point1])
                    while i < len(coords):
                        lineFrame = cv2.line(window_name, coords[i-1], coords[i], color, thick)
                        cv2.imshow('webcam', lineFrame)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(lineFrame,
                                    str(coords[i][0]) + ', ' + str(coords[i][1]),
                                    (coords[i][0] + 10, coords[i][1] - 20),
                                    font,
                                    0.6,
                                    color,
                                    1)
                        cv2.imshow('webcam', lineFrame)
                        i += 1
            else:
                reply = False
                cv2.imshow('webcam', frame)
            cv2.setMouseCallback('webcam', streamClicker)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #  final calculations and saving coordinates
                # DO NOT use filedialog in this function. There is a bug that will SEGFAULT the program
                savePath = messagebox.askyesno("Is this path okay?",
                                    "/home/andrew/Documents/digitizationTest")
                if savePath:
                    csvDigit = ah.csv_data_curveDigit(coords, "/home/andrew/Documents/digitizationTest")
                    if csvDigit == 1:
                        print("curve data saved")
                    else:
                        print("curve data failed to save")

                    reply = messagebox.askyesno("Finished?",
                                                "Continue choosing points?")
                if not reply:
                    setLoop = False
                    frame = None
                    time.sleep(2)
            # if cv2.waitKey(1) & 115 == ord('s'):
            #     csvDigit = ah.csv_data_curveDigit(coords, "/home/andrew/Documents/digitizationTest")
            #     if csvDigit == 1:
            #         print("Initial data saved")
            #     else:
            #         print("Initial data failed to save")
            #
            #     reply = messagebox.askyesno("Finished?",
            #                                 "Digitize another curve?"
            #                                 )
            #     if not reply:
            #         break

    # digitize curves until stopped by the user
    cv2.destroyWindow('webcam')
    time.sleep(1)


def streamClicker(event, x, y, flags, param):
    global frame, coords
    # get the center reference point
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        coords.append([x, y])
        # display coordinates on stream window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(x) + ', ' +
                    str(y), (x + 10, y - 20), font,
                    1, (250, 250, 250), 2)
        cv2.imshow('webcam', frame)
    if event == cv2.EVENT_RBUTTONDOWN:
        coords.pop()
    if event == cv2.EVENT_MOUSEMOVE:
        windSize = cv2.getWindowImageRect('webcam')
        window_name = frame
        start_point1 = (0, y)
        end_point1 = (windSize[2], y)
        start_point2 = (x, 0)
        end_point2 = (x, windSize[3])
        color = (250 , 250, 250)
        thick = 1
        cross1 = cv2.line(window_name, start_point1, end_point1, color, thick)
        cross2 = cv2.line(window_name, start_point2, end_point2, color, thick)
        cv2.imshow('webcam', cross1)
        cv2.imshow('webcam', cross2)


if __name__ == "__main__":
    '''
    Digitize curves from scanned plots
    '''

    # run the main function
    run()
    print('end')
