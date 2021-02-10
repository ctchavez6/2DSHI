import numpy as np
import cv2
from image_processing import bit_depth_conversion as bdc
from . import histograms as hgs
from PIL import Image, ImageDraw, ImageFont
from experiment_set_up import user_input_validation as uiv

"""


 _______________________________________________________
|                     Input Image                      |
|                      800 x 1200                      |
|                                                      |
|                      - - - -                         |
|                    /        \\                       | 800
|                   |   Beam   |                       |
|                   \\  - - - /                        |
|                                                      |
|                                                      |
|                                                      |
|______________________________________________________|
                       1200


                  _______________   
                 |               |                     
                 |     Static    |
                 |     Center    | 2*sigma_y
                 |     of Beam   |                     
                 |_______________|
                     2*sigma_x


Static center will be different for each camera, warp matrices try to get images as identical as possible.
"""

y_n_msg = "Proceed? (y/n): "
sixteen_bit_max = (2 ** 16) - 1
twelve_bit_max = (2 ** 12) - 1
eight_bit_max = (2 ** 8) - 1

def step_seven(stream, app, figs, histograms, lines, histograms_alg, lines_alg, figs_alg,
               histograms_r, lines_r, figs_r):
    last_frame = False

    desc = "Step 7 - Commence Image Algebra (Free Stream):"
    continue_stream = uiv.yes_no_quit(desc)
    print("You have entered step 7")
    s7_frame_count = 1
    frames_we_went_through = 0

    while continue_stream != last_frame:
        if last_frame:
            app.stop_streaming_override = False

        stream.frame_count += 1

        if stream.single_shot:

            print("Trigger for frame {0}".format(s7_frame_count))
            print("(If LAST FRAME, hit Toggle Button, THEN Trigger")
        #print("Step 7 Frame Count: ", s7_frame_count)
        #print("beginning")
        #print("\tcontinue_stream: ", continue_stream)
        #print("\tlast_frame: ", last_frame)
        #print("\tapp.stop_streaming_override: ", app.stop_streaming_override)

        stream.current_frame_a, stream.current_frame_b = stream.grab_frames(warp_matrix=stream.warp_matrix)
        
        x_a, y_a = stream.static_center_a
        x_b, y_b = stream.static_center_b

        # We initially set the LIMITS of the stream to 3 sigma
        # This is done so that if Stream Breaks when trying to show 3 sigma, we know something might be wrong?
        n_sigma = 3

        # Using array indexing to only keep relevant pixels
        stream.roi_a = stream.current_frame_a[
                     y_a - n_sigma * stream.static_sigmas_y: y_a + n_sigma * stream.static_sigmas_y + n_sigma,
                     x_a - n_sigma * stream.static_sigmas_x: x_a + n_sigma * stream.static_sigmas_x + n_sigma]

        stream.roi_b = stream.current_frame_b[
                     y_b - n_sigma * stream.static_sigmas_y: y_b + n_sigma * stream.static_sigmas_y + n_sigma,
                     x_b - n_sigma * stream.static_sigmas_x: x_b + n_sigma * stream.static_sigmas_x + n_sigma]


        roi_a = stream.roi_a
        b_double_prime = stream.roi_b
        CENTER_B_DP = int(b_double_prime.shape[1] * 0.5), int(b_double_prime.shape[0] * 0.5)


        x_a, y_a = CENTER_B_DP
        x_b, y_b = CENTER_B_DP
        n_sigma = app.foo
        #n_sigma = 0.75 #app.foo

        stream.roi_a = stream.roi_a[
                     int(y_a - n_sigma * stream.static_sigmas_y): int(y_a + n_sigma * stream.static_sigmas_y + 1),
                     int(x_a - n_sigma * stream.static_sigmas_x): int(x_a + n_sigma * stream.static_sigmas_x + 1)]

        stream.roi_b = b_double_prime[
                         int(y_b - n_sigma * stream.static_sigmas_y): int(y_b + n_sigma * stream.static_sigmas_y + 1),
                         int(x_b - n_sigma * stream.static_sigmas_x): int(x_b + n_sigma * stream.static_sigmas_x + 1)]

        if s7_frame_count == 0:
            print("stream.static_sigmas_y", stream.static_sigmas_y)
            print("stream.static_sigmas_x", stream.static_sigmas_x)
            print("stream.roi_a.shape", stream.roi_a.shape)
            print("stream.roi_b.shape", stream.roi_b.shape)


        h = stream.roi_b.shape[0]
        w = stream.roi_b.shape[1]

        hgs.update_histogram(histograms, lines, "a", 4096, stream.roi_a)
        hgs.update_histogram(histograms, lines, "b", 4096, stream.roi_b)
        figs["a"].canvas.draw()  # Draw updates subplots in interactive mode
        figs["b"].canvas.draw()  # Draw updates subplots in interactive mode
        hist_img_a = np.fromstring(figs["a"].canvas.tostring_rgb(), dtype=np.uint8, sep='')
        hist_img_b = np.fromstring(figs["b"].canvas.tostring_rgb(), dtype=np.uint8, sep='')  # convert  to image
        hist_img_a = hist_img_a.reshape(figs["a"].canvas.get_width_height()[::-1] + (3,))
        hist_img_b = hist_img_b.reshape(figs["b"].canvas.get_width_height()[::-1] + (3,))
        hist_img_a = cv2.resize(hist_img_a, (w, h), interpolation=cv2.INTER_AREA)
        hist_img_b = cv2.resize(hist_img_b, (w, h), interpolation=cv2.INTER_AREA)
        hist_img_a = bdc.to_16_bit(cv2.resize(hist_img_a, (w, h), interpolation=cv2.INTER_AREA), 8)
        hist_img_b = bdc.to_16_bit(cv2.resize(hist_img_b, (w, h), interpolation=cv2.INTER_AREA), 8)

        if s7_frame_count == 0:
            print("Stream seems to have error with the process below")
            print("ROI_A_WITH_HISTOGRAM = np.concatenate((cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR),"
                  " cv2.cvtColor(stream.roi_a * 16, cv2.COLOR_GRAY2BGR)),axis=1")
            print("First Step: np.concatenate requires both images to be the same shape. Let's check if they are")
            print("cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR).shape: ", cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR).shape)
            print("cv2.cvtColor(stream.roi_a * 16, cv2.COLOR_GRAY2BGR).shape: ", cv2.cvtColor(stream.roi_a * 16, cv2.COLOR_GRAY2BGR).shape)



        ROI_A_WITH_HISTOGRAM = np.concatenate(
            (cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR), cv2.cvtColor(stream.roi_a * 16, cv2.COLOR_GRAY2BGR)), axis=1)
        ROI_B_WITH_HISTOGRAM = np.concatenate(
            (cv2.cvtColor(hist_img_b, cv2.COLOR_RGB2BGR), cv2.cvtColor(stream.roi_b * 16, cv2.COLOR_GRAY2BGR)), axis=1)

        A_ON_B = np.concatenate((ROI_A_WITH_HISTOGRAM, ROI_B_WITH_HISTOGRAM), axis=0)

        plus_ = cv2.add(stream.roi_a, stream.roi_b)
        minus_ = np.zeros(stream.roi_a.shape, dtype='int16')
        minus_ = np.add(minus_, stream.roi_a)
        minus_ = np.add(minus_, stream.roi_b * (-1))
        # print("Lowest pixel in the minus spectrum: {}".format(np.min(minus_.flatten())))

        hgs.update_histogram(histograms_alg, lines_alg, "plus", 4096, plus_, plus=True)
        hgs.update_histogram(histograms_alg, lines_alg, "minus", 4096, minus_, minus=True)

        displayable_plus = cv2.add(stream.roi_a, stream.roi_b) * 16
        displayable_minus = cv2.subtract(stream.roi_a, stream.roi_b) * 16

        figs_alg["plus"].canvas.draw()  # Draw updates subplots in interactive mode
        hist_img_plus = np.fromstring(figs_alg["plus"].canvas.tostring_rgb(), dtype=np.uint8, sep='')
        hist_img_plus = hist_img_plus.reshape(figs_alg["plus"].canvas.get_width_height()[::-1] + (3,))
        hist_img_plus = cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA)
        hist_img_plus = bdc.to_16_bit(cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA), 8)
        PLUS_WITH_HISTOGRAM = np.concatenate(
            (cv2.cvtColor(hist_img_plus, cv2.COLOR_RGB2BGR), cv2.cvtColor(displayable_plus, cv2.COLOR_GRAY2BGR)),
            axis=1)

        figs_alg["minus"].canvas.draw()  # Draw updates subplots in interactive mode
        hist_img_minus = np.fromstring(figs_alg["minus"].canvas.tostring_rgb(), dtype=np.uint8,
                                       sep='')  # convert  to image
        hist_img_minus = hist_img_minus.reshape(figs_alg["minus"].canvas.get_width_height()[::-1] + (3,))
        hist_img_minus = cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA)
        hist_img_minus = bdc.to_16_bit(cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA), 8)
        MINUS_WITH_HISTOGRAM = np.concatenate(
            (cv2.cvtColor(hist_img_minus, cv2.COLOR_RGB2BGR), cv2.cvtColor(displayable_minus, cv2.COLOR_GRAY2BGR)),
            axis=1)

        ALGEBRA = np.concatenate((PLUS_WITH_HISTOGRAM, MINUS_WITH_HISTOGRAM), axis=0)
        DASHBOARD = np.concatenate((A_ON_B, ALGEBRA), axis=1)
        dash_height, dash_width, dash_channels = DASHBOARD.shape
        if dash_width > 2000:
            scale_factor = float(float(2000) / float(dash_width))
            DASHBOARD = cv2.resize(DASHBOARD, (int(dash_width * scale_factor), int(dash_height * scale_factor)))
        cv2.imshow("Dashboard", DASHBOARD)

        R_MATRIX = np.divide(minus_, plus_)
        nan_mean = np.nanmean(R_MATRIX.flatten())
        nan_st_dev = np.nanstd(R_MATRIX.flatten())

        DISPLAYABLE_R_MATRIX = np.zeros((R_MATRIX.shape[0], R_MATRIX.shape[1], 3), dtype=np.uint8)
        DISPLAYABLE_R_MATRIX[:, :, 1] = np.where(R_MATRIX < 0.00, abs(R_MATRIX * (2 ** 8 - 1)), 0)
        DISPLAYABLE_R_MATRIX[:, :, 2] = np.where(R_MATRIX < 0.00, abs(R_MATRIX * (2 ** 8 - 1)), 0)

        DISPLAYABLE_R_MATRIX[:, :, 2] = np.where(R_MATRIX > 0.00, abs(R_MATRIX * (2 ** 8 - 1)),
                                                 DISPLAYABLE_R_MATRIX[:, :, 2])

        dr_height, dr_width, dr_channels = DISPLAYABLE_R_MATRIX.shape

        hgs.update_histogram(histograms_r, lines_r, "r", 4096, R_MATRIX, r=True)
        figs_r["r"].canvas.draw()  # Draw updates subplots in interactive mode
        hist_img_r = np.fromstring(figs_r["r"].canvas.tostring_rgb(), dtype=np.uint8, sep='')  # convert  to image
        hist_img_r = hist_img_r.reshape(figs_r["r"].canvas.get_width_height()[::-1] + (3,))
        hist_img_r = cv2.resize(hist_img_r, (w, h), interpolation=cv2.INTER_AREA)
        hist_img_r = bdc.to_16_bit(cv2.resize(hist_img_r, (w, h), interpolation=cv2.INTER_AREA), 8)
        R_HIST = (cv2.cvtColor(hist_img_r, cv2.COLOR_RGB2BGR))

        R_VALUES = Image.new('RGB', (dr_width, dr_height), (eight_bit_max, eight_bit_max, eight_bit_max))

        draw = ImageDraw.Draw(R_VALUES)
        font = ImageFont.truetype('arial.ttf', size=30)
        (x, y) = (50, 50)
        message = "R Matrix Values\n"
        message = message + "Average: {0:.4f}".format(nan_mean) + "\n"
        message = message + "Sigma: {0:.4f}".format(nan_st_dev)

        # Mean: {0:.4f}\n".format(nan_mean, 2.000*float(stream.frame_count))
        color = 'rgb(0, 0, 0)'  # black color
        draw.text((x, y), message, fill=color, font=font)
        R_VALUES = np.array(R_VALUES)
        VALUES_W_HIST = np.concatenate((R_VALUES * (2 ** 8), np.array(R_HIST)), axis=1)

        cv2.imshow("R_MATRIX",
                   np.concatenate((VALUES_W_HIST, np.array(DISPLAYABLE_R_MATRIX * (2 ** 8), dtype='uint16')), axis=1))

        if last_frame:
            continue_stream = True
        else:
            continue_stream = stream.keep_streaming(one_by_one=True)

        if not continue_stream:
            if last_frame:
                pass
            else:
                if app is not None:
                    app.callback()

                cv2.destroyAllWindows()

        s7_frame_count += 1
        stream.R_HIST = R_HIST
        frames_we_went_through += 1

    cv2.destroyAllWindows()
    print("We completed this many frames: ", frames_we_went_through)
