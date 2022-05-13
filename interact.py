import os
import cv2
import numpy as np
import stroke_density, lighting_vectorized
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
 
 
def prepreprocess():
    scale_dict = {
        "sample-input.png": 100,
        "007.jpg": 60,
        "013.jpg": 100,
        "018.jpg": 60,
        "022.jpg": 60,
        "023.jpg": 30,
        "028.jpg": 40,
        "042.jpg": 60,
    }
    data_dir = "./imgs"
    filename = "023.jpg"
    scale_percent = scale_dict[filename]
    input_path = os.path.join(data_dir, filename)
    return input_path, scale_percent
 
 
def computeK(input_path, scale_percent):
    R = cv2.imread(input_path, cv2.IMREAD_COLOR)
    width = int(R.shape[1] * scale_percent / 100)
    height = int(R.shape[0] * scale_percent / 100)
    dim = (width, height)
    R = cv2.resize(R, dim, interpolation = cv2.INTER_AREA)
    K = stroke_density.get_stroke_density(R, input_path + "_intersections.npz")
    return R, K
 
 
def computeEK(R, K, which_corner):
    _, E = lighting_vectorized.get_lighting(R, which_corner, desat_coarse=True, coarse_desat_factor=0.5)
    EK = K + np.multiply(E, 1 - K).clip(0, 1)
    EK = EK.clip(0, 1)
    return EK
 
def computeI(EK, ambient_intensity, light_intensity):
    S = light_intensity * EK + ambient_intensity
    S = S.clip(0, 1)
    S = S**1.5
    I = np.multiply(R, S).clip(0, 255).astype(np.ubyte)
    b = I[:,:,0].copy()
    r = I[:,:,2].copy()
    I[:,:,0] = r
    I[:,:,2] = b
    return I
 
 
if __name__ == "__main__":
 
    input_path, scale_percent = prepreprocess()
    R, K = computeK(input_path, scale_percent)
 
    # build list of images in each corner
    eachCorner = []
    EKs = []
    for c in range(1, 5):
        EK = computeEK(R, K, c)
        EKs.append(EK)
        eachCorner.append(computeI(EK, 0, 1))
 
    img = plt.imshow(eachCorner[0])
    plt.axis("off")
 
    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)
 
    # slider that changes location of the light
    init_location = 0
    axlocation = plt.axes([0.25, 0.1, 0.65, 0.03])
    location_slider = Slider(
        ax=axlocation,
        label="Light Location",
        valmin=0,
        valmax=3,
        valfmt='%0.0f',
        valinit=init_location,
        color="silver"
    )
 
    # slider that changes red channel of light color
    init_r = 1
    axr = plt.axes([0.1, 0.25, 0.0225, 0.63])
    r_slider = Slider(
        ax=axr,
        label="R",
        valmin=0,
        valmax=1,
        valinit=init_r,
        orientation="vertical",
        color="salmon"
    )
 
    # slider that changes green channel of light color
    init_g = 1
    axg = plt.axes([0.15, 0.25, 0.0225, 0.63])
    g_slider = Slider(
        ax=axg,
        label="G",
        valmin=0,
        valmax=1,
        valinit=init_g,
        orientation="vertical",
        color="mediumseagreen"
    )
 
    # slider that changes blue channel of light color
    init_b = 1
    axb = plt.axes([0.2, 0.25, 0.0225, 0.63])
    b_slider = Slider(
        ax=axb,
        label="B",
        valmin=0,
        valmax=1,
        valinit=init_b,
        orientation="vertical",
        color="cornflowerblue"
    )
 
    # slider that changes blue channel of light color
    init_amb = 0
    axamb = plt.axes([0.25, 0.25, 0.0225, 0.63])
    amb_slider = Slider(
        ax=axamb,
        label="Amb",
        valmin=0,
        valmax=0.5,
        valinit=init_amb,
        orientation="vertical",
        color="silver"
    )
 
     # when location slider changes
    def location_update(val):
        img.set_data(eachCorner[int(round(location_slider.val))])
   
    def color_update(val):
        for c in range(4):
            eachCorner[c] = computeI(EKs[c], amb_slider.val, [b_slider.val, g_slider.val, r_slider.val])
        img.set_data(eachCorner[int(round(location_slider.val))])
 
    # update the image based on the slider changes
    location_slider.on_changed(location_update)
    r_slider.on_changed(color_update)
    b_slider.on_changed(color_update)
    g_slider.on_changed(color_update)
    amb_slider.on_changed(color_update)
    plt.show()
 
 
