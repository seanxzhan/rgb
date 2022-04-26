import cv2
import numpy as np
import stroke_density, lighting


def main():
    raw_img_path = "./data/sample-input.png"
    output_path = "./data/rendered.png"
    ambient_intensity = 0.55
    light_intensity = 1

    R = cv2.imread(raw_img_path, cv2.IMREAD_COLOR)
    R = lighting.pad_image(R)
    K = stroke_density.get_stroke_density(raw_img_path, "./data/stroke_density.png")
    K = np.repeat(K, 3, -1)
    cv2.imwrite("./data/K.png", (K).astype(np.ubyte))

    K = (K / 255).clip(0, 1)    # 0~1
    E = lighting.get_lighting(raw_img_path) # 0~1
    EK = np.multiply(E, K)
    # print("EK min: ", np.min(EK))
    # print("EK max: ", np.max(EK))
    cv2.imwrite("./data/EK.png", (EK * 255).astype(np.ubyte))
    S = light_intensity * EK + ambient_intensity
    S = S.clip(0, 1)
    I = np.multiply(R, S).clip(0, 255)

    cv2.imwrite(output_path, (I).astype(np.ubyte))


if __name__ == "__main__":
    main()


"""
R: original image
K: stroke density
E: coarse lighting effect
S: refined lighting
O: ambient intensity, default = 0.55
I: rendered result

- S = gamma * E • K + O
- I = R • S
"""
 