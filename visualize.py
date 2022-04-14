import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import trimesh


def show_image(img):
    # click on the image and press any key to exit
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def show_convex_hull(hull: ConvexHull):
    # note: point_color in hull = point_position
    print("visualizing convex hull...")

    # set up axes
    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # sample points on the hull's surface, show color
    mesh = trimesh.Trimesh(vertices=hull.points, faces=hull.simplices)
    sampled_points, _ = trimesh.sample.sample_surface(mesh, 10000)
    adj_col = sampled_points / 256
    adj_col = adj_col.tolist()
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], alpha=1, c=adj_col)

    return plt
