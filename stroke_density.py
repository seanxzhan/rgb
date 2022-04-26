import os
import numpy as np
import cv2
from scipy.spatial import ConvexHull
import visualize
import trimesh
from lighting import pad_image
from tqdm import tqdm


# find the points from an input image
def getPoints(image_string, pad=False):
    # cv2's channels are BGR
    img = cv2.imread(image_string, cv2.IMREAD_COLOR)
    if pad:
        img = pad_image(img)
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


def getMean(points):
    return np.mean(points, axis=0)


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
    # if False:
        print("loading intersection...")
        f = np.load(loc_out_path, allow_pickle=True)
        loc = f['loc']
        ray_idx = f['ray_idx']
    else:
        print("computing intersection...")
        # note: intersected location does follow the same order as ray_d!
        # each loc corresponds to each ray_idx, ray_idx indexes into ray_d
        loc, ray_idx, _ = mesh.ray.intersects_location(
            ray_origins=ray_p, ray_directions=ray_d)
        np.savez_compressed(loc_out_path, loc=loc, ray_idx=ray_idx)
    print("loaded intersection!")
    new_loc = np.zeros_like(loc)
    for i in range(loc.shape[0]):
        new_loc[ray_idx[i]] = loc[i]
    loc = new_loc
    # numerator = np.linalg.norm(hull.points - loc, axis=1, keepdims=True)
    numerator = np.linalg.norm(hull.points - ray_p, axis=1, keepdims=True)
    denominator = np.linalg.norm(ray_p - loc, axis=1, keepdims=True)
    K = numerator / denominator
    K *= 1.2
    K = np.clip(K, 0, 1) * 255
    return K

def save_stroke_density_as_img(stroke_density, h, w, out_path):
    print("saving stroke density to {} ...".format(out_path))
    img = np.reshape(stroke_density, (h, w, 1))
    K = img
    img = img.astype(np.uint8)
    cv2.imwrite(out_path, img)
    print("saved stroke density!")
    return K

def get_stroke_density(input_path, output_path, pad=False):
    points, H, W = getPoints(input_path, pad)
    hull = getHull(points)
    center = getMean(points)
    ray_dirs = getRayDirs(center, points)
    sd = getStrokeDensity(center, ray_dirs, hull, "./data/intersection.npz")
    return save_stroke_density_as_img(sd, H, W, output_path)


def get_all_stroke_density():
    data_dir = "./imgs"
    results_dir = "./stroke_density"
    for filename in tqdm(os.listdir("./imgs")):
        if os.path.splitext(filename)[1] != ".jpg":
            continue

        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(results_dir, filename)

        get_stroke_density(input_path, output_path, pad=True)


# run the stroke density algorithm
def main():
    # get_all_stroke_density()
    # exit(0)

    VIS = False
    USE_BARYCENTER = False

    input_filename = '013.jpg'

    # points, H, W = getPoints("./data/sample-input.png")
    points, H, W = getPoints("./imgs/" + input_filename)
    hull = getHull(points)
    if VIS:
        plt = visualize.show_convex_hull(hull)
        plt.savefig("./data/convex_hull_vis.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()
    
    if USE_BARYCENTER:
        center = getBarycenter(hull)
    else:
        center = getMean(points)

    ray_dirs = getRayDirs(center, points)
    stroke_density = getStrokeDensity(center, ray_dirs, hull, "./data/intersection.npz")
    # save_stroke_density_as_img(stroke_density, H, W, "./data/stroke_density.png")
    save_stroke_density_as_img(stroke_density, H, W, "./stroke_density/"+input_filename)


if __name__ == "__main__":
    main()
