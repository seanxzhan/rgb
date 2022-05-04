import wave
import cv2
import numpy as np
from tqdm import tqdm
import time

# channel intensity normalization (N)
# Input: rgb image [0, 255] (height, width, channels)
def computeNormalizedChannelIntensity(input):
    
    img_size = min([input.shape[0], input.shape[1]])
    kernel_size = int(img_size / 8)
    if (kernel_size % 2) == 0 :
        kernel_size = kernel_size + 1
    std_dev = round(kernel_size / 4)
    blurred_img = cv2.GaussianBlur(input, (kernel_size, kernel_size), std_dev, cv2.BORDER_DEFAULT)
    max_c = np.array([blurred_img[:, :, 0].max(), blurred_img[:, :, 1].max(), blurred_img[:, :, 2].max()])
    min_c = np.array([blurred_img[:, :, 0].min(), blurred_img[:, :, 1].min(), blurred_img[:, :, 2].min()])
    normalized_c = (blurred_img - min_c) / (max_c - min_c).astype(np.float64)

    return normalized_c

# vectorized interpolation of yinterp/xinterp arrays given image to interpolate
def bilinear_interpolate(image, yinterp, xinterp, height, width):

    y0 = np.floor(yinterp).clip(0, height - 2).astype(int)
    y1 = y0 + 1
    x0 = np.floor(xinterp).clip(0, width - 2).astype(int)
    x1 = x0 + 1
    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]
    wa = ((x1-xinterp) * (y1-yinterp)).reshape(height,width,1)
    wb = ((x1-xinterp) * (yinterp-y0)).reshape(height,width,1)
    wc = ((xinterp-x0) * (y1-yinterp)).reshape(height,width,1)
    wd = ((xinterp-x0) * (yinterp-y0)).reshape(height,width,1)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


# coarse lighting effect
# Input: N - normalized input channels
# Params: L - light position, delta - scaling scalar, distance offset
def computeCoarseLightingEffect(N, which_corner):

    # find the coordinates of the light given the corner input
    height, width, _  = N.shape
    if which_corner == 1:
        # top right
        orig_l_x = l_x = width - 1
        orig_l_y = l_y = 0
    elif which_corner == 2:
        # bottom right
        orig_l_x = l_x = width - 1
        orig_l_y = l_y = height - 1
    elif which_corner == 3:
        # bottom left
        orig_l_x = l_x = 0
        orig_l_y = l_y = height - 1
    elif which_corner == 4:
        # top left
        orig_l_x = l_x = 0
        orig_l_y = l_y = 0
   
    # large delta gives better coarse lighting results, small delta gives better final results
    delta = 0.01

    # determine light/mouse location between [-1,1]
    l_x = (2 * l_x / float(width)) - 1
    l_y = (2 * l_y / float(height)) - 1
    l_z = 1

    # find px, py, pd
    p_x, p_y = np.meshgrid((2 * np.arange(width) / float(width)) - 1, (2 * np.arange(height) / float(height)) - 1)
    p_d = np.sqrt((p_x - l_x)**2 + (p_y - l_y)**2)

    # find sin_theta and cos theta given px, py, pd
    sin_theta = np.divide(p_y - l_y, p_d + 1e-10)
    cos_theta = np.divide(p_x - l_x, p_d + 1e-10)

    # remove infinte sines and cosines
    if (orig_l_x >= 0 and orig_l_x < width and orig_l_y >= 0 and orig_l_y < height):
        sin_theta[orig_l_y, orig_l_x] = 0
        cos_theta[orig_l_y, orig_l_x] = 1

    # calculate light direction
    l_z_array = np.full((height, width), l_z)
    lights_direction = np.array((l_z_array, p_d)).transpose().swapaxes(0,1)
    lights_direction /= np.linalg.norm(lights_direction, axis=2).reshape(height, width, 1)
    lights_direction = np.reshape(lights_direction, (height, width, 1, 2))

    # calculate wave direction
    n1y = (((l_y + sin_theta * p_d) + 1) * height / 2).round().astype(int)
    n1x = (((l_x + cos_theta * p_d) + 1) * width / 2).round().astype(int)
    n1 = N[n1y, n1x]
    yinterp = (((l_y + sin_theta * (p_d + delta)) + 1) * height / 2).clip(1e-5, height - 1 - 1e-5)
    xinterp = (((l_x + cos_theta * (p_d + delta)) + 1) * width / 2).clip(1e-5, width - 1 - 1e-5)
    n2 = bilinear_interpolate(N, yinterp, xinterp, height, width)
    delta_array = np.full((height, width, 3), delta)
    waves_direction = np.array((n2 - n1, delta_array)).transpose().swapaxes(0,2)
    waves_direction /= np.linalg.norm(waves_direction, axis=3).reshape(height, width, 3, 1)
    
    # calculate E
    E = np.sum(lights_direction * waves_direction, axis=3).clip(0,1)
    return E


# pads array to desired shape with pad_value
def to_shape(a, shape, pad_value):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant',
                  constant_values = pad_value)


# pads image (using background) to be square
def pad_image(img):

    padding_value_0 = img[0, 0, 0] 
    padding_value_1 = img[0, 0, 1] 
    padding_value_2 = img[0, 0, 2] 
    max_side_length = max(img.shape[0], img.shape[1])
    padded_img = np.stack((
        to_shape(img[:, :, 0], (max_side_length, max_side_length), padding_value_0),
        to_shape(img[:, :, 1], (max_side_length, max_side_length), padding_value_1),
        to_shape(img[:, :, 2], (max_side_length, max_side_length), padding_value_2)
    ), axis = 2)

    return padded_img


def get_lighting(img, which_corner):

    N = computeNormalizedChannelIntensity(img)
    E = computeCoarseLightingEffect(N, which_corner)
    return N, E


def main():

    start_time = time.time()
    print("Testing lighting.py")
    img = cv2.imread("./tmp/sample-input.png", cv2.IMREAD_COLOR)
    N = computeNormalizedChannelIntensity(img)
    cv2.imwrite("./tmp/N.png", (N * 255).astype(np.ubyte))
    which_corner = 1
    E = computeCoarseLightingEffect(N, which_corner)
    cv2.imwrite("./tmp/E"+str(which_corner)+".png", (E * 255).astype(np.ubyte))
    print("time taken = " + str(round(time.time() - start_time, 3)))


if __name__ == "__main__":
    main()