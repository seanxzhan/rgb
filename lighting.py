import cv2
import numpy as np
from tqdm import tqdm

# channel intensity normalization (N)
# Input: rgb image [0, 255] (height, width, channels)
def computeNormalizedChannelIntensity(input):
    
    # I made kernel_size and std_dev for kernel relative to image size
    img_size = min([input.shape[0], input.shape[1]])
    # img_size = max([input.shape[0], input.shape[1]])
    
    kernel_size = int(img_size / 8)
    # Make odd sized
    if (kernel_size % 2) == 0 :
        kernel_size = kernel_size + 1

    std_dev = round(kernel_size / 4)

    blurred_img = cv2.GaussianBlur(input, (kernel_size, kernel_size), std_dev, cv2.BORDER_DEFAULT)
    max_c = np.array([blurred_img[:, :, 0].max(), blurred_img[:, :, 1].max(), blurred_img[:, :, 2].max()])
    min_c = np.array([blurred_img[:, :, 0].min(), blurred_img[:, :, 1].min(), blurred_img[:, :, 2].min()])

    normalized_c = (blurred_img - min_c) / (max_c - min_c).astype(np.float64)
    # normalized_c = blurred_img / max_c

    return normalized_c


def bilinear_interpolate(im, x, y, width, height):
    epsilon = 1e-5
    if (x < 0): x = epsilon
    if (y < 0): y = epsilon
    if (x >= (width - 1)): x = width - 1 - epsilon
    if (y >= (height - 1)): y = height - 1 - epsilon

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

# coarse lighting effect
# Input: N - normalized input channels
# Params: L - light position, delta - scaling scalar, distance offset
def computeCoarseLightingEffect(N):
    # TODO: Vectorize this
   
    # Top right of image
    # # orig_l_x = l_x = 480
    # # orig_l_y = l_y = 0
    # # orig_l_x = l_x = 480
    # # orig_l_y = l_y = 680
    # # orig_l_x = l_x = 0
    # # orig_l_y = l_y = 680
    # orig_l_x = l_x = 0
    # orig_l_y = l_y = 0
    # orig_l_x = l_x = 550
    # orig_l_y = l_y = 0
    # orig_l_x = l_x = 550
    # orig_l_y = l_y = 600
    # orig_l_x = l_x = 0
    # orig_l_y = l_y = 600
    orig_l_x = l_x = 0
    orig_l_y = l_y = 0
    l_z = 1

    height, width, _  = N.shape
    print(width, height)

    top_left = np.array([0, 0], dtype=np.float32)
    top_right = np.array([0, width-1], dtype=np.float32)
    bottom_left = np.array([height-1, 0], dtype=np.float32)
    bottom_right = np.array([height-1, width-1], dtype=np.float32)
    light_loc = np.array([l_y, l_x], dtype=np.float32)
    corners = [top_left, top_right, bottom_left, bottom_right]
    dists = []
    for corner in corners:
        dists.append(np.sqrt(np.square(light_loc - corner)))
    max_dist = np.max(dists)
   
    # Is this a relative or absolute value??
    # delta = 2 * float(1.0) / float(width) # This is 1 pixel in [-1, 1] x [-1, 1]
    # delta = 2 * 50 * float(1.0) / float(width) # This works much better 
    delta = 0.01

    # Normalizing light/mouse location to [-1, 1] x [-1, 1]
    # I think l_z = 1 is fine??
    l_x = (2 * l_x / float(width)) - 1
    l_y = (2 * l_y / float(height)) - 1
    print(l_x, l_y)

    p_x, p_y = np.meshgrid((2 * np.arange(width) / float(width)) - 1, (2 * np.arange(height) / float(height)) - 1)
    p_d = np.sqrt((p_x - l_x)**2 + (p_y - l_y)**2)
    
    sin_theta = np.divide(p_y - l_y, p_d)
    cos_theta = np.divide(p_x - l_x, p_d)

    # If Light is above image, need to replace sin and cos at point
    # Exact direction shouldn't matter, because infinite choices
    # Otherwise, sin and cos will be inf or cause nan later on
    if (orig_l_x >= 0 and orig_l_x < width and  orig_l_y >= 0 and orig_l_y < height):
        sin_theta[orig_l_y, orig_l_x] = 0
        cos_theta[orig_l_y, orig_l_x] = 1

    E = np.empty(N.shape)
    for y in tqdm(range(height)):
        for x in range(width):
            for c in range(3):
                light_direction = np.array([l_z, p_d[y, x]])
                light_direction = light_direction / np.sqrt(np.sum(light_direction**2))
                
                # n1 - Represents origin point on N 
                n1 = N[
                    round(((l_y + sin_theta[y, x] * p_d[y, x]) + 1) * height / float(2)),
                    round(((l_x + cos_theta[y, x] * p_d[y, x]) + 1) * width / float(2)), 
                    c]

                assert(abs(((l_y + sin_theta[y, x] * p_d[y, x]) + 1) * height / float(2) - y) < 1e3)
                assert(abs(((l_x + cos_theta[y, x] * p_d[y, x]) + 1) * width / float(2) - x) < 1e3)

                x_interp = ((l_x + cos_theta[y, x] * (p_d[y, x] + delta)) + 1) * width / float(2)
                y_interp = ((l_y + sin_theta[y, x] * (p_d[y, x] + delta)) + 1) * height / float(2)

                # n2 - Interpolated point on N
                n2 = bilinear_interpolate(N[:, :, c], x_interp, y_interp, width, height)

                wave_direction = np.array([n2 - n1, delta])
                wave_direction = wave_direction / np.sqrt(np.sum(wave_direction**2))

                E[y, x, c] = np.dot(wave_direction, light_direction)
                # if (E[y, x, c] < 0):
                #     print("bad")
                #     print(E[y, x, c])
    E = np.clip(E, 0, 1)
    return E

# refined lighting effect

# Pads array to desired shape with pad_value
def to_shape(a, shape, pad_value):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant',
                  constant_values = pad_value)

# Pads image (using background) to be square
def pad_image(img):
    
    # Assuming corner is background for our padding colors for now
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

def get_lighting(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    # padded_img = pad_image(img)
    N = computeNormalizedChannelIntensity(img)
    cv2.imwrite("./data/normalized-channels.png", (N * 255).astype(np.ubyte))
    E = computeCoarseLightingEffect(N)
    print(E.min(), E.max())
    cv2.imwrite("./data/E.png", (E * 255).astype(np.ubyte))
    return E

def main():
    print("Testing lighting.py")
    img = cv2.imread("./data/013.jpg", cv2.IMREAD_COLOR)
    # padded_img = pad_image(img)

    N = computeNormalizedChannelIntensity(img)
    cv2.imwrite("./data/normalized-channels.png", (N * 255).astype(np.ubyte))

    # Uses saved N image from paper screenshot
    # N = cv2.imread("./data/sample-N.png", cv2.IMREAD_COLOR)
    # N = pad_image(N)
    # N = N / float(255) 

    E = computeCoarseLightingEffect(N)
    # print(E.min(), E.max())
    cv2.imwrite("./data/E4.png", (E * 255).astype(np.ubyte))

if __name__ == "__main__":
    main()