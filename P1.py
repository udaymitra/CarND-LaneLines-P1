import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

### My code

def read_image_from_path(img_file_path):
    return mpimg.imread(img_file_path)

def write_image_to_path(img, target_file_path):
    mpimg.imsave(target_file_path, img)

def get_polygon_mask(shape):
    maxY, maxX = shape
    leftBottom = (int(0.1 * maxX), maxY)
    leftTop = (int(0.45 * maxX), int(0.6 * maxY))
    rightTop = (int(0.55 * maxX), int(0.6 * maxY))
    rightBottom = (int(0.9 * maxX), maxY)

    # leftBottom = (0, 539)
    # leftTop =
    return np.array([[leftBottom, leftTop, rightTop, rightBottom]], dtype=np.int32)


from scipy.optimize import curve_fit


def linear_fit_func(x, m, b):
    return m * x + b


def fit_line(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    params = curve_fit(linear_fit_func, x, y)
    [m, b] = params[0]
    return [m, b]

def fit_line_from_segments(segments):
    points = []
    for line in segments:
        for x1, y1, x2, y2 in line:
            points.append([x1, y1])
            points.append([x2, y2])

            # slope = (y2 - y1) / (x2 - x1)
            # b = y2 - (slope * x2)
            # for x in range(x1, x2, 5):
            #     y = slope * x + b
            #     points.append([x, y])

    return fit_line(points)

def filter_lane_segments_and_split(lines, image_shape):
    """
    This method takes in hough lines identified using cvs.HoughLinesP method
    Computes slope for each segment
    For both left and right lane segments, we have a range of acceptable slopes
    Split segments into left and right lane segments
    When slope doesnt fall in these ranges, filter out the line
    """
    left_lane_slope_limits = [0.5, 2.5]
    right_lane_slope_limits = [-2.5, -0.5]
    midx = int(image_shape[1] / 2)

    left_lane_lines = []
    right_lane_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y1 - y2) / (x2 - x1)  # y goes 0 - N top down
            if slope > left_lane_slope_limits[0] and slope < left_lane_slope_limits[1] and x1 <= midx and x2 <= midx:
                left_lane_lines.append(line)
            elif slope > right_lane_slope_limits[0] and slope < right_lane_slope_limits[1] and x1 > midx and x2 > midx:
                right_lane_lines.append(line)

    return [left_lane_lines, right_lane_lines]


def extrapolate_lane_segments_and_merge(left_lane_lines, right_lane_lines, shape_of_original_image):
    """
    method to merge different segments of left and right lanes

    step 1: curve fit all left lane points to get best line that describes all left lane segments.
            this gives us (m_left, b_left)
    step 2: curve fit all right lane points to get best line  that describes all right lane segments.
            this gives us (m_right, b_right)
    step 4: we have bottom of image (ybot) and middle (ymid) (using mask). get xbot, xmid
            - by using (m_left and b_left) for left lane
            - by using (m_right and b_right) for left lane
            Now, we have 2 points that represent left lane and 2 points represent right lane
    step 5: return 2 lines computed in step 4
    """
    [m_left, b_left] = fit_line_from_segments(left_lane_lines)
    [m_right, b_right] = fit_line_from_segments(right_lane_lines)

    maxY = shape_of_original_image[0]
    bottom_of_mask = maxY
    top_of_mask = int(0.6 * maxY)

    left_lane_bottom = [int((bottom_of_mask - b_left) / m_left), bottom_of_mask]
    left_lane_top = [int((top_of_mask - b_left) / m_left), top_of_mask]
    right_lane_bottom = [int((bottom_of_mask - b_right) / m_right), bottom_of_mask]
    right_lane_top = [int((top_of_mask - b_right) / m_right), top_of_mask]

    out_lines = np.ndarray(shape=(2, 1, 4), dtype=np.int32)
    out_lines[0] = (left_lane_bottom[0], left_lane_bottom[1], left_lane_top[0], left_lane_top[1])
    out_lines[1] = (right_lane_bottom[0], right_lane_bottom[1], right_lane_top[0], right_lane_top[1])

    return out_lines


def get_hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def draw_line_image_hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = get_hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)
    filtered_lines = filter_lane_segments_and_split(lines, img.shape)
    # merged_lane_lines = extrapolate_lane_segments_and_merge(filtered_lines[0], filtered_lines[1], img.shape)
    all_filtered_lines = filtered_lines[0] + filtered_lines[1]
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, all_filtered_lines, thickness=5)
    return line_img

def get_lanes(img, mask_function):
    # convert to gray scale
    gray = grayscale(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # plt.imshow(gray, cmap='hsv')
    # plt.show()

    # canny edge detection
    # blur image
    blurred_gray = gaussian_blur(gray, 5)

    # run canny
    lines_in_img = canny(blurred_gray, 50, 150)

    # compute region mask
    mask_vertices = mask_function(lines_in_img.shape)

    # apply mask
    lines_in_roi = region_of_interest(lines_in_img, mask_vertices)

    # hough transform
    rho = 2
    theta = np.pi/180
    threshold = 30
    min_line_length = 10
    max_line_gap = 5
    lane_lines = draw_line_image_hough_lines(lines_in_roi, rho, theta, threshold, min_line_length, max_line_gap)

    return lane_lines

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    lane_lines = get_lanes(image, get_polygon_mask)
    img_with_lanes = weighted_img(lane_lines, image)
    return img_with_lanes

# white_output = 'white.mp4'
# clip1 = VideoFileClip("solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)
#
# yellow_output = 'yellow.mp4'
# clip2 = VideoFileClip('solidYellowLeft.mp4')
# yellow_clip = clip2.fl_image(process_image)
# yellow_clip.write_videofile(yellow_output, audio=False)

# import os
# for f in os.listdir("challenge_debug/frames/"):
#     img = read_image_from_path("challenge_debug/frames/" + f)
#     img_with_lanes = process_image(img)
#     write_image_to_path(img_with_lanes, "challenge_debug/output/" + f)

# challenge_output = 'extra.mp4'
# clip2 = VideoFileClip('challenge.mp4')
# challenge_clip = clip2.fl_image(process_image)
# challenge_clip.write_videofile(challenge_output, audio=False)

def apply_white_yellow_hsv_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([130, 130, 255])
    # Threshold the HSV image to get only white colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # define range of yellow color in HSV
    lower_yellow = np.array([20, 80, 200])
    upper_yellow = np.array([120, 200, 255])
    # Threshold the HSV image to get only yellow colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = mask_yellow | mask_white

    # Bitwise-AND mask and original image
    return cv2.bitwise_and(image, one_frame, mask=mask)

one_frame = read_image_from_path("challenge_debug/frames/output_0041.jpg")
masked_frame = apply_white_yellow_hsv_mask(one_frame)
plt.imshow(masked_frame)
plt.show()