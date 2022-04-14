import os
import numpy as np
import cv2
from scipy.spatial import ConvexHull
import visualize
import trimesh


# find the points from an input image
def getPoints(image_string):
    # cv2's channels are BGR
    img = cv2.imread(image_string, cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    # flip the r and b channels
    points = np.reshape(img, [-1, 3])
    r = points[:, 0].copy()
    b = points[:, 2].copy()
    points[:, 0] = b
    points[:, 2] = r
    return points, height, width


# build the convex hull in RGB space
def getHull(points):
    hull = ConvexHull(points)
    return hull


# find the barycenter of the convex hull
def getBarycenter(hull):
    points = hull.points
    faces = hull.simplices
    surface_area = hull.area
    v0 = points[faces[:,0]]
    v1 = points[faces[:,1]]
    v2 = points[faces[:,2]]
    centroid = (v0 + v1 + v2)/3
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).reshape(-1,1)
    total = np.sum(centroid * area, axis=0)/surface_area
    return total


def getRayDirs(barycenter, points):
    diff = points - barycenter
    factor = np.linalg.norm(diff, axis=1, keepdims=True)
    return diff / factor


# get the intersections of each ray from the barycenter to the colors
def getStrokeDensity(barycenter, ray_d, hull, loc_out_path):
    num_rays = ray_d.shape[0]
    ray_p = np.repeat(np.reshape(barycenter, (1,-1)), axis=0, repeats=num_rays)
    mesh = trimesh.Trimesh(vertices=hull.points, faces=hull.simplices)
    if os.path.exists(loc_out_path):
        print("loading intersection...")
        loc = np.load(loc_out_path, allow_pickle=True)['loc']
    else:
        print("computing intersection...")
        loc, _, _ = mesh.ray.intersects_location(
            ray_origins=ray_p, ray_directions=ray_d)
        np.savez_compressed(loc_out_path, loc=loc)
    numerator = np.linalg.norm(hull.points - loc, axis=1, keepdims=True)
    denominator = np.linalg.norm(ray_p - loc, axis=1, keepdims=True)
    K = numerator / denominator
    K = np.abs(1 - K)
    K = np.clip(K, 0, 1) * 255
    return K


def save_stroke_density_as_img(stroke_density, h, w, out_path):
    print("saving stroke density to {} ...".format(out_path))
    img = np.reshape(stroke_density, (h, w, 1))
    img = img.astype(np.uint8)
    cv2.imwrite(out_path, img)


# run the stroke density algorithm
def main():
    VIS = False

    points, H, W = getPoints("./data/sample-input.png")
    hull = getHull(points)
    if VIS:
        plt = visualize.show_convex_hull(hull)
        plt.savefig("./data/convex_hull_vis.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()
    barycenter = getBarycenter(hull)
    ray_dirs = getRayDirs(barycenter, points)
    stroke_density = getStrokeDensity(barycenter, ray_dirs, hull, "./data/intersection.npz")
    save_stroke_density_as_img(stroke_density, H, W, "./data/stroke_density.png")



if __name__ == "__main__":
    main()
