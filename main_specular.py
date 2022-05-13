import os
import re
import cv2
import numpy as np
from PIL import Image
import stroke_density, lighting_specular


def main(input_path, depth_in_path, results_dir,
         light_which_corner, eye_which_corner):
    raw_img_path = input_path
    output_path = os.path.join(results_dir, "rendered"+str(light_which_corner)+".png")
    output_w_spec_path = os.path.join(results_dir, "rendered"+str(light_which_corner)+str(eye_which_corner)+".png")
    K_out = os.path.join(results_dir, "K.png")
    intersect_path = os.path.join(results_dir, "intersections.npz")
    N_out = os.path.join(results_dir, "N.png")
    E_out = os.path.join(results_dir, "E" + str(light_which_corner) + ".png")
    EK_out = os.path.join(results_dir, "EK" + str(light_which_corner) + ".png")
    S_out = os.path.join(results_dir, "S" + str(light_which_corner) + ".png")
    SS_out = os.path.join(results_dir, "S" + str(light_which_corner) + str(eye_which_corner) + ".png")
    blurred_depth_out = os.path.join(results_dir, "blurred_depth.png")
    norm_out = os.path.join(results_dir, "normals.png")
    specular_out = os.path.join(results_dir, "spec" + str(eye_which_corner) + ".png")

    ambient_intensity = 0
    light_intensity = 1

    # original image
    R = cv2.imread(raw_img_path, cv2.IMREAD_COLOR)
    R = lighting_specular.resize_and_pad(R)

    # stroke density
    K = stroke_density.get_stroke_density(R, intersect_path)
    cv2.imwrite(K_out, (K * 255).astype(np.ubyte))

    # get depth
    d_im = cv2.imread(depth_in_path, cv2.IMREAD_GRAYSCALE)
    d_im = d_im[:, int(d_im.shape[1] / 2):]

    # normalized channel and coarse lighting
    d_im, normals, N, E, specular =\
        lighting_specular.get_lighting(R, d_im, light_which_corner, eye_which_corner)

    cv2.imwrite(blurred_depth_out, d_im)
    cv2.imwrite(norm_out, (normals * 255).astype(np.ubyte))
    cv2.imwrite(N_out, (N * 255).astype(np.ubyte))
    cv2.imwrite(E_out, (E * 255).astype(np.ubyte))
    cv2.imwrite(specular_out, (specular * 255).astype(np.ubyte))

    # refinement
    EK = K + np.multiply(E, 1 - K).clip(0, 1)
    EK = EK.clip(0, 1)
    cv2.imwrite(EK_out, (EK * 255).astype(np.ubyte))

    # output, normal
    S = light_intensity * EK + ambient_intensity
    S = S.clip(0, 1)
    S = S**1.5
    I = np.multiply(R, S).clip(0, 255)
    cv2.imwrite(S_out, (S * 255).astype(np.ubyte))
    cv2.imwrite(output_path, (I).astype(np.ubyte))

    # output, specular
    SS = light_intensity * EK + ambient_intensity + specular
    SS = SS.clip(0, 1)
    SS = SS**1.5
    II = np.multiply(R, SS).clip(0, 255)
    cv2.imwrite(SS_out, (SS * 255).astype(np.ubyte))
    cv2.imwrite(output_w_spec_path, (II).astype(np.ubyte))


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
    id = "012"
    assert id in ["007", "012", "013", "016", "028", "cb"]

    data_dir = "./imgs"
    depth_dir = "./specular"
    all_results_dir = "./results"
    filename = id+".jpg"

    input_path = os.path.join(data_dir, filename)
    depth_input_path = os.path.join(depth_dir, id, "padded_R_depth.jpg")
    results_dir = os.path.join(all_results_dir, id + "_spclr")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # set light to top right
    # for all 4 eye corners
    for c in range(1, 5):
        main(input_path, depth_input_path, results_dir,
             light_which_corner=c, eye_which_corner=3)

    def condition(f):
        return f[:8] == "rendered"

    save_images_as_gif(results_dir, condition, 'paint.gif')
