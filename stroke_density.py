import numpy as np
import cv2
from scipy.spatial import ConvexHull


# find the points from an input image
def getPoints(image_string):
    img = cv2.imread(image_string, cv2.IMREAD_COLOR)
    num_rows, num_cols, num_channels = img.shape
    points = []
    for i in range(num_rows):
        for j in range(num_cols):
            points.append(img[i,j].tolist())
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
    total = [0,0,0]
    for face in faces:
        v0 = points[face[0]]
        v1 = points[face[1]]
        v2 = points[face[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        centroid = (v0 + v1 + v2)/3
        total += area * centroid
    total /= surface_area
    return total


# calculate the outgoing rays for 
def getRays(barycenter, points):
    rays = []
    for point in points:
        ray = (point - barycenter)/np.linalg.norm(point - barycenter)
        rays.append(ray)
    return rays


# get the intersections of each ray from the barycenter to the colors
def getIntersections(rays, hull):
    return []


# run the stroke density algorithm
def main():
    points = getPoints("./data/sample-input.png")
    hull = getHull(points)
    barycenter = getBarycenter(hull)
    rays = getRays(barycenter, points)
    intersections = getIntersections(rays, hull)


if __name__ == "__main__":
    main()
