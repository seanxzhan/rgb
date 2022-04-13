# FEATURES:
#   (DONE) load image
#   button that allows you to load any image
#   show a circle + dot diagram like the paper
#   arrow keys control the position of the light, the dot in the cirle updates accordingly
#   slider to control the ambient light intensity
#   (DONE) slider to control light intensity
#   (DONE) sliders to control light color
#   (DONE) reset button to bring sliders back to original positions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# define initial values for parameters
init_r = 255
init_g = 255
init_b = 255
init_lamda = 0.5

# create the initial image
loaded_image = plt.imread("./data/sample-input.png")
data = loaded_image
img = plt.imshow(data)
plt.axis("off")

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# slider that changes intensity of light
axlamda = plt.axes([0.25, 0.1, 0.65, 0.03])
lamda_slider = Slider(
    ax=axlamda,
    label='Intensity',
    valmin=0,
    valmax=1,
    valinit=init_lamda,
    color="silver"
)

# slider that changes red channel of light color
axr = plt.axes([0.1, 0.25, 0.0225, 0.63])
r_slider = Slider(
    ax=axr,
    label="R",
    valmin=0,
    valmax=255,
    valinit=init_r,
    valfmt='%0.0f',
    orientation="vertical",
    color="salmon"
)

# slider that changes green channel of light color
axg = plt.axes([0.15, 0.25, 0.0225, 0.63])
g_slider = Slider(
    ax=axg,
    label="G",
    valmin=0,
    valmax=255,
    valinit=init_g,
    valfmt='%0.0f',
    orientation="vertical",
    color="mediumseagreen"
)

# slider that changes blue channel of light color
axb = plt.axes([0.2, 0.25, 0.0225, 0.63])
b_slider = Slider(
    ax=axb,
    label="B",
    valmin=0,
    valmax=255,
    valinit=init_b,
    valfmt='%0.0f',
    orientation="vertical",
    color="cornflowerblue"
)

# The function to be called anytime a slider's value changes
def update(val):
    data[:,:,0] = r_slider.val/255
    data[:,:,1] = g_slider.val/255
    data[:,:,2] = b_slider.val/255
    img.set_data(data)

# update the image based on the slider changes
r_slider.on_changed(update)
g_slider.on_changed(update)
b_slider.on_changed(update)
lamda_slider.on_changed(update)

# create button that resets sliders when clicked
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    r_slider.reset()
    g_slider.reset()
    b_slider.reset()
    lamda_slider.reset()
button.on_clicked(reset)

plt.show()