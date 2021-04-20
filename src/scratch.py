import numpy as np
import cv2

blk_image = np.zeros([300, 300, 3])
blk_image2 = cv2.ellipse(blk_image.copy(), (150, 150), 10,
                         angle, startAngle, endAngle, color, thickness)

combined = blk_image2[:, :, 0] + blk_image2[:, :, 1] + blk_image2[:, :, 2]
rows, cols = np.where(combined > 0)
