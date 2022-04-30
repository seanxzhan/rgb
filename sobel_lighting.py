import cv2
import numpy as np
from lighting import pad_image
import stroke_density as sd

def generate_lighting_effects(stroke_density, content):

    # perform gaussian kernel
    blur = cv2.GaussianBlur(content, (5, 5), 1, cv2.BORDER_DEFAULT)
    c = cv2.filter2D(blur, cv2.CV_32F, np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]))
    r = cv2.filter2D(blur, cv2.CV_32F, np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]]))

    # normalization
    EPS = 1e-10
    max_effect = np.max((c**2 + r**2)**0.5)
    c = (c + EPS) / (max_effect + EPS)
    r = (r + EPS) / (max_effect + EPS)

    # refinement
    stroke_density_scaled = (stroke_density.astype(np.float32) / 255.0).clip(0, 1)
    c *= (1.0 - stroke_density_scaled ** 2.0 + EPS) ** 0.5
    r *= (1.0 - stroke_density_scaled ** 2.0 + EPS) ** 0.5
    refined_result = np.stack([stroke_density_scaled, r, c], axis=2)
    return refined_result

# Then generate the lighting effects
def main():

    # back lighting bug parameters
    # light_color = np.array([1, 1, 1])
    # light_location = [1, 50, 50]
    # ambient_intensity = 0.1
    # light_intensity = 1

    # find the raw image and stroke density maps
    # raw_image = cv2.imread("./data/padded_sample_input.png", cv2.IMREAD_COLOR)
    # stroke_density = cv2.imread("./data/stroke_density.png", cv2.IMREAD_GRAYSCALE)
    raw_image = cv2.imread("./imgs/sample-input.png", cv2.IMREAD_COLOR)
    height, width, _ = raw_image.shape
    stroke_density = sd.get_stroke_density(raw_image, './tmp/intersection.npz')
    stroke_density = stroke_density[:, :, 0]
    stroke_density *= 255

    which_corner = 1

    # define variables
    if which_corner == 1:
        # top right
        light_x = width - 1
        light_y = 0
    elif which_corner == 2:
        # bottom right
        light_x = width - 1
        light_y = height - 1
    elif which_corner == 3:
        # bottom left
        light_x = 0
        light_y = height - 1
    elif which_corner == 4:
        # top left
        light_x = 0
        light_y = 0

    light_color = np.array([1, 1, 1])
    light_location = [
        1,
        - light_y / height * 2 + 1,
        - light_x / width * 2 + 1]
    print(light_location)
    ambient_intensity = 0.45
    light_intensity = 1

    # generate lighting affects
    lighting_effect = np.stack([
        generate_lighting_effects(stroke_density, raw_image[:, :, 0]),
        generate_lighting_effects(stroke_density, raw_image[:, :, 1]),
        generate_lighting_effects(stroke_density, raw_image[:, :, 2])
    ], axis=2)
    
    # compute dot product to calculate final effect
    light_source_location = np.array([[light_location]], dtype=np.float32)
    light_source_direction = light_source_location / np.sqrt(np.sum(np.square(light_source_location)))
    final_effect = np.sum(lighting_effect * light_source_direction, axis=3).clip(0, 1)
    cv2.imwrite("./tmp/sobel_effect.png", (final_effect * 255).astype(np.ubyte))

    # compute new image from the final effect, raw image, and other parameters
    rendered_image = ((ambient_intensity + final_effect * light_intensity) * light_color * raw_image).clip(0,255).astype(np.uint8)
    cv2.imwrite("./tmp/E1.png", (rendered_image).astype(np.ubyte))


if __name__ == "__main__":
    main()