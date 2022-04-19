import wave
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
    # l_x = 475
    # l_y = 50
    # l_z = 1
    l_x = 480
    l_y = 0
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
   
    delta = float(10)

    p_x, p_y = np.meshgrid(np.arange(width), np.arange(height))
    p_d = np.sqrt((p_x - l_x)**2 + (p_y - l_y)**2)

    light_dir_p_d = p_d / max_dist

    sin_theta = np.divide(p_y - l_y, p_d)
    cos_theta = np.divide(p_x - l_x, p_d)

    if (l_x >= 0 and l_x < width and  l_y >= 0 and l_y < height):
        sin_theta[l_y, l_x] = 0
        cos_theta[l_y, l_x] = 1
    
    counter = 0
    E = np.empty(N.shape)
    for y in tqdm(range(height)):
        for x in range(width):
            for c in range(3):
                light_direction = np.array([l_z, light_dir_p_d[y, x]])
                light_direction = light_direction / np.linalg.norm(light_direction)
                
                # Represent origin point on N and interpolated point offset by delta
                n1 = N[
                    round((l_y + sin_theta[y, x] * p_d[y, x])),
                    round((l_x + cos_theta[y, x] * p_d[y, x])), 
                    c]

                x_interp = l_x + cos_theta[y, x] * (p_d[y, x] + delta)
                y_interp = l_y + sin_theta[y, x] * (p_d[y, x] + delta) 

                n2 = N[
                    np.clip(round(y_interp), 0, height-1),
                    np.clip(round(x_interp), 0, width-1), 
                    c]

                # n2 = bilinear_interpolate(N[:, :, c], x_interp, y_interp, width, height)

                # print("y: ", y)
                # print("x: ", x)
                # print("y_interp: ", y_interp)
                # print("x_interp: ", x_interp)
                # print("n1: ", n1)
                # print("n2: ", n2)

                wave_direction = np.array([n2 - n1, delta])
                # print(wave_direction)
                wave_direction = wave_direction / np.linalg.norm(wave_direction)
                # print(wave_direction)

                E[y, x, c] = np.dot(wave_direction, light_direction)
                # print(E[y, x, c])

                counter += 1
    E = np.clip(E, 0, 1)
    return E

# refined lighting effect

def main():
    print("Testing lighting.py")
    img = cv2.imread("./data/sample-input.png", cv2.IMREAD_COLOR)

    N = computeNormalizedChannelIntensity(img)
    cv2.imwrite("./data/normalized-channels.png", (N * 255).astype(np.ubyte))

    # TODO: push to branch
    # TODO: examine mouse/light position bounds in Painting Light code
    N = cv2.imread("./data/normalized-channels.png", cv2.IMREAD_COLOR)
    N = N / float(255) 

    E = computeCoarseLightingEffect(N)
    cv2.imwrite("./data/coarse-lighting-effect.png", (E * 255).astype(np.ubyte))

if __name__ == "__main__":
    main()