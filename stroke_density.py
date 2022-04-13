import numpy as np
import cv2
from scipy.spatial import ConvexHull


# find the points from an input image
def getPoints(image_string):
    img = cv2.imread(image_string, cv2.IMREAD_COLOR)
    num_rows, num_cols, num_channels = img.shape
    points = img.reshape(num_rows*num_cols,num_channels)
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


# calculate rays between the image colors and the barycenter
def getRays(barycenter, points):
    difference = points - barycenter
    norm = np.linalg.norm(difference, axis=1).reshape(-1,1)
    rays = difference/(norm)
    return rays


# get the intersections of each ray from the barycenter to the colors
def getIntersections(rays, hull):
    return []


# run the stroke density algorithm
def main():
    points = getPoints("./data/sample-input.png")
    hull = getHull(points)
    barycenter = getBarycenter(hull)
    rays = getRays(barycenter, hull.points)
    intersections = getIntersections(rays, hull)


if __name__ == "__main__":
    main()
