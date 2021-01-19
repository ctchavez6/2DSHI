import numpy as np
import os
import matplotlib.pyplot as plt
from image_processing import bit_depth_conversion as bdc
from PIL import Image, ImageDraw, ImageFont
from . import histograms as hgs
import cv2

y_n_msg = "Proceed? (y/n): "
sixteen_bit_max = (2 ** 16) - 1
twelve_bit_max = (2 ** 12) - 1
eight_bit_max = (2 ** 8) - 1


def step_eight(stream, run_folder, app, figs, histograms, lines, histograms_alg, lines_alg, figs_alg,
               histograms_r, lines_r, figs_r):
    record_r_matrices = input("Step 8 - Image Algebra (Record): Proceed? (y/n): ")
    satisfied_with_run = False

    while satisfied_with_run is False:

        current_r_frame = 0
        stream.stats = list()
        stream.r_frames = list()
        stream.a_frames = list()
        stream.b_prime_frames = list()

        stream.a_images = list()
        stream.b_prime_images = list()

        stream.stats.append(["Frame", "Avg_R", "Sigma_R"])

        # r_matrix_limit = int(input("R Matrix Frame Break: "))
        if record_r_matrices.lower() == "y":
            continue_stream = True
            while continue_stream:
                stream.frame_count += 1
                stream.current_frame_a, stream.current_frame_b = stream.grab_frames(warp_matrix=stream.warp_matrix)
                current_r_frame += 1
                print("Current R Frame: {}".format(current_r_frame))

                x_a, y_a = stream.static_center_a
                x_b, y_b = stream.static_center_b

                n_sigma = 3

                stream.roi_a = stream.current_frame_a[
                             y_a - n_sigma * stream.static_sigmas_y: y_a + n_sigma * stream.static_sigmas_y + n_sigma,
                             x_a - n_sigma * stream.static_sigmas_x: x_a + n_sigma * stream.static_sigmas_x + n_sigma]

                stream.roi_b = stream.current_frame_b[
                             y_b - n_sigma * stream.static_sigmas_y: y_b + n_sigma * stream.static_sigmas_y + n_sigma,
                             x_b - n_sigma * stream.static_sigmas_x: x_b + n_sigma * stream.static_sigmas_x + n_sigma]

                if stream.warp_matrix_2 is not None:
                    roi_a, b_double_prime = stream.grab_frames2(stream.roi_a.copy(), stream.roi_b.copy(),
                                                              stream.warp_matrix_2.copy())
                else:
                    roi_a, b_double_prime = stream.grab_frames2(stream.roi_a.copy(), stream.roi_b.copy(),
                                                              stream.warp_matrix.copy())

                CENTER_B_DP = int(b_double_prime.shape[1] * 0.5), int(b_double_prime.shape[0] * 0.5)

                x_a, y_a = CENTER_B_DP
                x_b, y_b = CENTER_B_DP
                n_sigma = app.foo

                stream.a_frames.append(roi_a)
                stream.b_prime_frames.append(b_double_prime)
                stream.a_images.append(roi_a)
                stream.b_prime_images.append(b_double_prime)

                stream.roi_a = stream.roi_a[
                             int(y_a - n_sigma * stream.static_sigmas_y): int(
                                 y_a + n_sigma * stream.static_sigmas_y + 1),
                             int(x_a - n_sigma * stream.static_sigmas_x): int(
                                 x_a + n_sigma * stream.static_sigmas_x + 1)]

                b_double_prime = b_double_prime[
                                 int(y_b - n_sigma * stream.static_sigmas_y): int(
                                     y_b + n_sigma * stream.static_sigmas_y + 1),
                                 int(x_b - n_sigma * stream.static_sigmas_x): int(
                                     x_b + n_sigma * stream.static_sigmas_x + 1)]

                stream.roi_b = b_double_prime
                h = b_double_prime.shape[0]
                w = b_double_prime.shape[1]

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

                ROI_A_WITH_HISTOGRAM = np.concatenate(
                    (cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR), cv2.cvtColor(stream.roi_a * 16, cv2.COLOR_GRAY2BGR)),
                    axis=1)
                ROI_B_WITH_HISTOGRAM = np.concatenate(
                    (cv2.cvtColor(hist_img_b, cv2.COLOR_RGB2BGR), cv2.cvtColor(stream.roi_b * 16, cv2.COLOR_GRAY2BGR)),
                    axis=1)

                A_ON_B = np.concatenate((ROI_A_WITH_HISTOGRAM, ROI_B_WITH_HISTOGRAM), axis=0)

                plus_ = cv2.add(stream.roi_a, stream.roi_b)
                minus_ = np.zeros(stream.roi_a.shape, dtype='int16')
                minus_ = np.add(minus_, stream.roi_a)
                minus_ = np.add(minus_, stream.roi_b * (-1))

                hgs.update_histogram(histograms_alg, lines_alg, "plus", 4096, plus_, plus=True)
                hgs.update_histogram(histograms_alg, lines_alg, "minus", 4096, minus_, minus=True)

                displayable_plus = cv2.add(stream.roi_a, stream.roi_b) * 16
                displayable_minus = cv2.subtract(stream.roi_a, stream.roi_b) * 16

                figs_alg["plus"].canvas.draw()  # Draw updates subplots in interactive mode
                hist_img_plus = np.fromstring(figs_alg["plus"].canvas.tostring_rgb(), dtype=np.uint8, sep='')
                hist_img_plus = hist_img_plus.reshape(figs_alg["plus"].canvas.get_width_height()[::-1] + (3,))
                hist_img_plus = cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA)
                hist_img_plus = bdc.to_16_bit(cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA), 8)
                PLUS_WITH_HISTOGRAM = np.concatenate((cv2.cvtColor(hist_img_plus, cv2.COLOR_RGB2BGR),
                                                      cv2.cvtColor(displayable_plus, cv2.COLOR_GRAY2BGR)), axis=1)

                figs_alg["minus"].canvas.draw()  # Draw updates subplots in interactive mode
                hist_img_minus = np.fromstring(figs_alg["minus"].canvas.tostring_rgb(), dtype=np.uint8,
                                               sep='')  # convert  to image
                hist_img_minus = hist_img_minus.reshape(figs_alg["minus"].canvas.get_width_height()[::-1] + (3,))
                hist_img_minus = cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA)
                hist_img_minus = bdc.to_16_bit(cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA), 8)
                MINUS_WITH_HISTOGRAM = np.concatenate((cv2.cvtColor(hist_img_minus, cv2.COLOR_RGB2BGR),
                                                       cv2.cvtColor(displayable_minus, cv2.COLOR_GRAY2BGR)), axis=1)

                ALGEBRA = np.concatenate((PLUS_WITH_HISTOGRAM, MINUS_WITH_HISTOGRAM), axis=0)
                DASHBOARD = np.concatenate((A_ON_B, ALGEBRA), axis=1)
                dash_height, dash_width, dash_channels = DASHBOARD.shape

                if dash_width > 2000:
                    scale_factor = float(float(2000) / float(dash_width))
                    DASHBOARD = cv2.resize(DASHBOARD, (int(dash_width * scale_factor), int(dash_height * scale_factor)))

                cv2.imshow("Dashboard", DASHBOARD)

                R_MATRIX = np.divide(minus_, plus_)
                stream.r_frames.append(R_MATRIX)
                nan_mean = np.nanmean(R_MATRIX.flatten())
                nan_st_dev = np.nanstd(R_MATRIX.flatten())
                stream.stats.append([len(stream.r_frames), nan_mean, nan_st_dev])

                DISPLAYABLE_R_MATRIX = np.zeros((R_MATRIX.shape[0], R_MATRIX.shape[1], 3), dtype=np.uint8)
                DISPLAYABLE_R_MATRIX[:, :, 1] = np.where(R_MATRIX < 0.00, abs(R_MATRIX * (2 ** 8 - 1)), 0)
                DISPLAYABLE_R_MATRIX[:, :, 2] = np.where(R_MATRIX < 0.00, abs(R_MATRIX * (2 ** 8 - 1)), 0)

                DISPLAYABLE_R_MATRIX[:, :, 2] = np.where(R_MATRIX > 0.00, abs(R_MATRIX * (2 ** 8 - 1)),
                                                         DISPLAYABLE_R_MATRIX[:, :, 2])

                dr_height, dr_width, dr_channels = DISPLAYABLE_R_MATRIX.shape

                hgs.update_histogram(histograms_r, lines_r, "r", 4096, R_MATRIX, r=True)
                figs_r["r"].canvas.draw()  # Draw updates subplots in interactive mode
                hist_img_r = np.fromstring(figs_r["r"].canvas.tostring_rgb(), dtype=np.uint8,
                                           sep='')  # convert  to image
                hist_img_r = hist_img_r.reshape(figs_r["r"].canvas.get_width_height()[::-1] + (3,))
                hist_img_r = cv2.resize(hist_img_r, (w, h), interpolation=cv2.INTER_AREA)
                hist_img_r = bdc.to_16_bit(cv2.resize(hist_img_r, (w, h), interpolation=cv2.INTER_AREA), 8)

                stream.R_HIST = (cv2.cvtColor(hist_img_r, cv2.COLOR_RGB2BGR))

                R_VALUES = Image.new('RGB', (dr_width, dr_height), (eight_bit_max, eight_bit_max, eight_bit_max))

                # initialise the drawing context with
                # the image object as background

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
                VALUES_W_HIST = np.concatenate((R_VALUES * (2 ** 8), np.array(stream.R_HIST)), axis=1)

                cv2.imshow("R_MATRIX", np.concatenate(
                    (VALUES_W_HIST, np.array(DISPLAYABLE_R_MATRIX * (2 ** 8), dtype='uint16')), axis=1))

                continue_stream = stream.keep_streaming()
                if continue_stream is False:
                    satisfied_with_range = False
                    while satisfied_with_range is False:
                        cv2.destroyAllWindows()
                        fig_ = plt.figure()
                        ax1 = fig_.add_subplot(111)
                        frames = list()
                        averages = list()
                        sigmas = list()

                        starting_frame = int(input("Start at frame: "))
                        end_frame = int(input("End at frame: "))

                        # for i in range(1, len(stats)):
                        for i in range(starting_frame, end_frame + 1):
                            frames.append(stream.stats[i][0])
                            averages.append(stream.stats[i][1])
                            sigmas.append(stream.stats[i][2])

                        ax1.errorbar(frames, averages, yerr=sigmas, c='b', capsize=5)
                        ax1.set_xlabel('Frame')
                        ax1.set_ylabel('R (Mean)')
                        ax1.set_title('Mean R by Frame')
                        ax1.axhline(y=-1.0, xmin=starting_frame, xmax=end_frame)
                        ax1.axhline(y=0.0, xmin=starting_frame, xmax=end_frame)
                        ax1.axhline(y=1.0, xmin=starting_frame, xmax=end_frame)

                        save_path = os.path.join(run_folder, 'mean_r_by_frame.png')
                        fig_.savefig(save_path)
                        plot_img = cv2.imread(save_path, cv2.IMREAD_COLOR)
                        cv2.imshow('R Mean Plot', plot_img)
                        cv2.waitKey(60000)
                        cv2.destroyAllWindows()
                        range_satisfaction_input = input("Are you satisfied with this range? (y/n): ")
                        if range_satisfaction_input.lower() == "y":
                            satisfied_with_range = True
                            stream.start_writing_at = starting_frame
                            stream.end_writing_at = end_frame
                    satisfaction_input = input("Are you satisfied with this run? (y/n): ")
                    if satisfaction_input.lower() == 'y':
                        satisfied_with_run = True
        if record_r_matrices.lower() == "n":
            satisfied_with_run = True
            continue_stream = False
