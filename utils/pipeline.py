import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import pickle
from utils.helper import *

class Pipeline:
    def __init__(self, mtx, dist, sx_thresh, dir_thresh, s_thresh):
        self.mtx = mtx
        self.dist = dist
        self.sx_thresh = sx_thresh
        self.dir_thresh = dir_thresh
        self.s_thresh = s_thresh
        self.detected = False
        self.left_fit = None
        self.right_fit = None


    def pipeline(self, img):
        dst_img = dis_cor(img, self.mtx, self.dist)

        offset = 300
        w, h = dst_img.shape[1], dst_img.shape[0]
        dst = np.float32([[offset, 0], [w - offset, 0], [offset, h], [w - offset, h]])
        src = np.float32([(582, 460), (705, 460), (260, 680), (1050, 680)])
        warped = perspective_transform(dst_img, src, dst)

        warped = grad_col_threshold(warped, self.sx_thresh, self.dir_thresh, self.s_thresh)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        if self.detected:
            left_fit, right_fit, left_curverad, right_curverad, left_fitx, right_fitx, ploty = search_around_poly(
                warped, self.left_fit, self.right_fit, ym_per_pix, xm_per_pix)
            self.left_fit = left_fit
            self.right_fit = right_fit

        else:
            # use slide windows
            out_img, left_fit, right_fit, left_curverad, right_curverad, left_fitx, right_fitx, ploty = fit_polynomial(
                warped, ym_per_pix, xm_per_pix)
            self.left_fit = left_fit
            self.right_fit = right_fit

        # check if the line is detected
        if abs(left_curverad - right_curverad) > 300:
            self.detected = False
        else:
            self.detected = True

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        # newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

        newwarp = perspective_transform(color_warp, dst, src)
        # Combine the result with the original image
        result = cv2.addWeighted(cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB), 1, newwarp, 0.3, 0)

        font = cv2.FONT_HERSHEY_DUPLEX
        text = 'Curve radius: ' + '{:2.2f}'.format((left_curverad + right_curverad) / 2000) + 'km'
        cv2.putText(result, text, (40, 70), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)

        # get shift from center
        histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        center = (leftx_base + rightx_base) / 2
        shift = (center - midpoint) * xm_per_pix
        if shift <= 0:
            text2 = 'Shift ' + '{:2.2f}'.format(-shift) + 'm to left'
        else:
            text2 = 'Shift ' + '{:2.2f}'.format(shift) + 'm to right'
        cv2.putText(result, text2, (40, 150), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result
