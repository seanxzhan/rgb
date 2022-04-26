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

def canvas_to_norm(height, width, y, x):
    # maps canvas coords to [-0.5, 0.5]
    h_c = int(height / 2)
    w_c = int(width / 2)

    adj_y = - (y - h_c) / height
    adj_x = (x - w_c) / width

    return adj_y, adj_x

def norm_to_canvas(height, width, y, x):
    # maps norm coords to canvas
    h_c = int(height / 2)
    w_c = int(width / 2)

    adj_y = - height * y + h_c
    adj_x = width * x + w_c

    return adj_y, adj_x

def computeCoarseLightingEffect(N):
    delta_pd = 0.06
    light_x = 0
    light_y = 0
    light_z = 1

    height, width, _ = N.shape
    light_y, light_x = canvas_to_norm(height, width, light_y, light_x)

    E = np.empty(N.shape)
    for orig_y in tqdm(range(height)):
        for orig_x in range(width):
            for c in range(3):
                y, x = canvas_to_norm(height, width, orig_y, orig_x)

                pd = np.sqrt((light_y - y)**2 + (light_x - x)**2)
                sin_theta = (y - light_y) / pd
                cos_theta = (x - light_x) / pd

                light_dir = np.array([light_z, pd], dtype=np.float32)
                light_dir = light_dir / np.linalg.norm(light_dir)

                n_pd_x = light_x + pd * cos_theta
                n_pd_y = light_y + pd * sin_theta
                canvas_y, canvas_x = norm_to_canvas(height, width, n_pd_y, n_pd_x)
                n_pd = bilinear_interpolate(
                    N[:, :, c], canvas_x, canvas_y, width, height)

                n_delta_pd_x = light_x + (pd + delta_pd) * cos_theta
                n_delta_pd_y = light_y + (pd + delta_pd) * sin_theta
                canvas_y, canvas_x = norm_to_canvas(height, width, n_delta_pd_y, n_delta_pd_x)
                n_dealta_pd = bilinear_interpolate(
                    N[:, :, c], canvas_x, canvas_y, width, height)

                wave_dir_x = n_dealta_pd - n_pd
                wave_dir_y = delta_pd
                wave_dir = np.array([wave_dir_x, wave_dir_y], dtype=np.float32)
                wave_dir = wave_dir / np.linalg.norm(wave_dir)

                e = light_dir[0] * wave_dir[0] + light_dir[1] * wave_dir[1]
                E[orig_y, orig_x, c] = e
    E = np.clip(E, 0, 1)
    return E

if __name__ == "__main__":
    img = cv2.imread("./data/sample-input.png", cv2.IMREAD_COLOR)
    N = computeNormalizedChannelIntensity(img)
    cv2.imwrite("./data/N.png", (N * 255).astype(np.ubyte))
    E = computeCoarseLightingEffect(N)
    cv2.imwrite("./data/E4.png", (E * 255).astype(np.ubyte))
