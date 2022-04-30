import os
import re
import cv2
import numpy as np
from PIL import Image
import stroke_density, lighting


def main(input_path, results_dir, scale_percent, which_corner):
    raw_img_path = input_path
    output_path = os.path.join(results_dir, "rendered"+str(which_corner)+".png")
    K_out = os.path.join(results_dir, "K.png")
    intersect_path = os.path.join(results_dir, "intersections.npz")
    N_out = os.path.join(results_dir, "N.png")
    E_out = os.path.join(results_dir, "E" + str(which_corner) + ".png")
    EK_out = os.path.join(results_dir, "EK" + str(which_corner) + ".png")
    S_out = os.path.join(results_dir, "S" + str(which_corner) + ".png")

    ambient_intensity = 0
    light_intensity = 1

    # original image
    R = cv2.imread(raw_img_path, cv2.IMREAD_COLOR)
    print("original shape: {}, {}".format(R.shape[1], R.shape[0]))
    width = int(R.shape[1] * scale_percent / 100)
    height = int(R.shape[0] * scale_percent / 100)
    dim = (width, height)
    R = cv2.resize(R, dim, interpolation = cv2.INTER_AREA)
    print("resized shape: {}, {}".format(height, width))
    
    # stroke density
    K = stroke_density.get_stroke_density(R, intersect_path)
    cv2.imwrite(K_out, (K * 255).astype(np.ubyte))

    # normalized channel and coarse lighting
    N, E = lighting.get_lighting(R, which_corner) # 0~1
    cv2.imwrite(N_out, (N * 255).astype(np.ubyte))
    cv2.imwrite(E_out, (E * 255).astype(np.ubyte))

    # TODO: single image normal estimation for specular

    # refinement
    EK = K + np.multiply(E, 1 - K).clip(0, 1)
    EK = EK.clip(0, 1)
    cv2.imwrite(EK_out, (EK * 255).astype(np.ubyte))

    # output
    S = light_intensity * EK + ambient_intensity
    S = S.clip(0, 1)
    S = S**1.5
    cv2.imwrite(S_out, (S * 255).astype(np.ubyte))
    I = np.multiply(R, S).clip(0, 255)
    cv2.imwrite(output_path, (I).astype(np.ubyte))


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def save_images_as_gif(dir_, condition, gif_name):
    frames = []
    imgs = sorted_alphanumeric(os.listdir(dir_))
    for i in imgs:
        if condition(i):
            new_frame = Image.open(os.path.join(dir_, i))
            frames.append(new_frame)

    out_gif = os.path.join(dir_, gif_name)
    frames[0].save(out_gif, format='GIF', append_images=frames[1:],
                    save_all=True, duration=300, loop=0)
    print("saved to:", out_gif)


if __name__ == "__main__":
    scale_dict = {
        "sample-input.png": 100,
        "007.jpg": 60,
        "013.jpg": 60,
        "018.jpg": 60,
        "022.jpg": 60,
        "028.jpg": 40,
        "042.jpg": 60,
    }

    data_dir = "./imgs"
    all_results_dir = "./results"
    filename = "028.jpg"

    # scales down the image to make things go faster
    # manually add key value pair above
    scale_percent = scale_dict[filename]

    input_path = os.path.join(data_dir, filename)
    results_dir = os.path.join(all_results_dir, os.path.splitext(filename)[0])
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # for all 4 corners
    for c in range(1, 5):
        main(input_path, results_dir, scale_percent, c)

    def condition(f):
        return f[:8] == "rendered"

    save_images_as_gif(results_dir, condition, 'paint.gif')
