import numpy as np
import cv2
from scipy.spatial import ConvexHull
import visualize


# find the points from an input image
def getPoints(image_string):
    # cv2's channels are BGR
    img = cv2.imread(image_string, cv2.IMREAD_COLOR)
    # flip the r and b channels
    points = np.reshape(img, [-1, 3])
    r = points[:, 0].copy()
    b = points[:, 2].copy()
    points[:, 0] = b
    points[:, 2] = r
    return points


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
def getIntersections(barycenter, rays, hull):
    return []


# run the stroke density algorithm
def main():
    points = getPoints("./data/sample-input.png")
    hull = getHull(points)
    visualize.show_convex_hull(hull)
    barycenter = getBarycenter(hull)
    ray_dirs = getRayDirs(barycenter, points)
    intersections = getIntersections(barycenter, ray_dirs, hull)


if __name__ == "__main__":
    main()
