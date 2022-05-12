from socket import AF_IPX
import cv2
from matplotlib.pyplot import axis
import numpy as np
from lighting_vectorized import computeNormalizedChannelIntensity, bilinear_interpolate


def resize_and_pad(R):
    print("orig shape: ", R.shape)
    height, width, _ = R.shape
    ratio = width / height

    # target_height = 480
    # target_width = 640
    target_height = 288
    target_width = 512
    adj_width = int(ratio * target_height)

    dim = (adj_width, target_height)
    R = cv2.resize(R, dim, interpolation = cv2.INTER_AREA)
    # print("resized shape: ", R.shape)

    # ignoring even / odd right now
    left = int((target_width - adj_width) / 2)
    right = target_width - left - adj_width
    R = cv2.copyMakeBorder(R, 0, 0, left, right, cv2.BORDER_REPLICATE)
    # print("padded shape: ", R.shape)

    print("resized and padded shape: ", R.shape)

    return R


def get_normal(d_im):
    zy, zx = np.gradient(d_im)
    # zx = cv2.Sobel(d_im, cv2.CV_32F, 1, 0, ksize=3)
    # zy = cv2.Sobel(d_im, cv2.CV_32F, 0, 1, ksize=3)

    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255

    return normal


def computeCoarseLightingEffect(N, normals, light_which_corner, eye_which_corner):

    # find the coordinates of the light given the corner input
    height, width, _  = N.shape
    if light_which_corner == 1:
        # top right
        orig_l_x = l_x = width - 1
        orig_l_y = l_y = 0
    elif light_which_corner == 2:
        # bottom right
        orig_l_x = l_x = width - 1
        orig_l_y = l_y = height - 1
    elif light_which_corner == 3:
        # bottom left
        orig_l_x = l_x = 0
        orig_l_y = l_y = height - 1
    elif light_which_corner == 4:
        # top left
        orig_l_x = l_x = 0
        orig_l_y = l_y = 0

    if eye_which_corner == 1:
        eye_x = width - 1
        eye_y = 0
    elif eye_which_corner == 2:
        eye_x = width - 1
        eye_y = height - 1
    elif eye_which_corner == 3:
        eye_x = 0
        eye_y = height -1 
    elif eye_which_corner == 4:
        eye_x = 0
        eye_y = 0
    eye_z = 1

    eye_y = (2 * eye_y / height) - 1
    eye_x = (2 * eye_x / width) - 1
   
    # large delta gives better coarse lighting results, small delta gives better final results
    delta = 0.1

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
    
    # compute pixel to light dir
    all_pix_x = np.reshape(p_x, (-1, 1, 1))
    all_pix_y = np.reshape(p_y, (-1, 1, 1))
    all_pix_x = np.repeat(all_pix_x, 3, axis=-1)
    all_pix_y = np.repeat(all_pix_y, 3, axis=-1)
    all_pix_z = np.reshape(n1, (-1, 1, 3))
    pix_pos = np.concatenate((all_pix_x, all_pix_y, all_pix_z), axis=1)
    # print("pix pos shape: ", pix_pos.shape)
    # print(pix_pos[0, :, 0])

    light_pos = np.array([[l_x, l_y, l_z]], dtype=np.float32)
    light_pos = np.repeat(light_pos, pix_pos.shape[0], axis=0)
    light_pos = np.expand_dims(light_pos, axis=-1)
    light_pos = np.repeat(light_pos, 3, axis=-1)
    # print("light pos shape: ", light_pos.shape)
    # print(light_pos[0, :, 0])

    eye_pos = np.array([[eye_x, eye_y, eye_z]], dtype=np.float32)
    eye_pos = np.repeat(eye_pos, pix_pos.shape[0], axis=0)
    eye_pos = np.expand_dims(eye_pos, axis=-1)
    eye_pos = np.repeat(eye_pos, 3, axis=-1)
    # print("eye pos shape: ", eye_pos.shape)
    # print(eye_pos[0, :, 0])

    n = np.reshape(normals, (-1, 3))
    n = np.expand_dims(n, axis=-1)
    n = np.repeat(n, 3, axis=-1)
    # print("normals shape: ", n.shape)
    # print(n[0, :, 0])

    w = pix_pos - light_pos
    nnn = np.linalg.norm(w, axis=1, keepdims=True)
    w = w / (nnn + 1e-5)

    dott = np.sum(w * n, axis=1, keepdims=True)
    dott = np.clip(dott, 0, 1)
    w_r = w - 2 * dott * n
    nnn = np.linalg.norm(w_r, axis=1, keepdims=True)
    w_r = w_r / (nnn + 1e-5)

    e = eye_pos - pix_pos
    nnn = np.linalg.norm(e, axis=1, keepdims=True)
    e = e / (nnn + 1e-5)

    dott = np.sum(w_r * e, axis=1, keepdims=True)
    dott = np.clip(dott, 0, 1)

    k_s = 1
    expo = 10
    light = np.array([1, 1, 1], dtype=np.float32)
    light = np.reshape(light, (1, 1, 3))
    specular = light * k_s * (dott**expo)
    specular = np.reshape(specular, (height, width, -1))
    print("specular shape: ", specular.shape)

    # calculate E
    E = np.sum(lights_direction * waves_direction, axis=3).clip(0,1)
    return E, specular


def get_lighting(R, d_im, light_which_corner, eye_which_corner):
    # img should already be resized and padded
    d_im = cv2.blur(d_im, (20, 20))
    norm = get_normal(d_im)
    h, w, _ = norm.shape
    norm /= 255
    norm = np.reshape(norm, (-1, 3))
    norm = norm / np.linalg.norm(norm, axis=-1, keepdims=True)
    norm = np.reshape(norm, (h, w, 3))
    N = computeNormalizedChannelIntensity(R)
    E, specular = computeCoarseLightingEffect(
        N, norm, light_which_corner, eye_which_corner)
    return d_im, norm, N, E, specular


if __name__ == "__main__":
    id = "007"
    assert id in ["007", "012", "013", "016", "028", "cb"]
    R = cv2.imread("./imgs/"+id+".jpg", cv2.IMREAD_COLOR)
    R = resize_and_pad(R)
    cv2.imwrite("./specular/"+id+"/padded_R.png", R.astype(np.ubyte))

    d_im = cv2.imread("./specular/"+id+"/padded_R_depth.jpg",
                      cv2.IMREAD_GRAYSCALE)
    d_im = d_im[:, int(d_im.shape[1] / 2):]

    d_im = cv2.blur(d_im, (20, 20))
    cv2.imwrite("./specular/"+id+"/blurred_depth.png", d_im)

    norm = get_normal(d_im)
    h, w, _ = norm.shape
    norm /= 255
    norm = np.reshape(norm, (-1, 3))
    norm = norm / np.linalg.norm(norm, axis=-1, keepdims=True)
    norm = np.reshape(norm, (h, w, 3))
    cv2.imwrite("./specular/"+id+"/normal.png", norm*255)

    N = computeNormalizedChannelIntensity(R)
    cv2.imwrite("./specular/"+id+"/N.png", (N * 255).astype(np.ubyte))
    light_which_corner = 1
    eye_which_corner = 2
    E, specular = computeCoarseLightingEffect(
        N, norm, light_which_corner, eye_which_corner)
    cv2.imwrite(
        "./specular/"+id+"/specular"+str(light_which_corner)+str(eye_which_corner)+".png",
        (specular * 255).astype(np.ubyte))
