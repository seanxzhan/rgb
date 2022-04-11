import cv2
from cv2 import blur
import numpy as np

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

    # print(max_c)
    # print(min_c)

    normalized_c = (blurred_img - min_c) / (max_c - min_c).astype(np.float64)

    return normalized_c

# coarse lighting effect

# refined lighting effect

def main():
    print("Testing lighting.py")
    img = cv2.imread("./data/sample-input.png", cv2.IMREAD_COLOR)
    N = computeNormalizedChannelIntensity(img)
    N = (N * 255).astype(np.ubyte)
    cv2.imwrite("./data/normalized-channels.png", N)

if __name__ == "__main__":
    main()