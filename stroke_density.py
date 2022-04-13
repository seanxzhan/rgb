import cv2
from scipy.spatial import ConvexHull


# build the convex hull in RGB space
def getHull(points):
    hull = ConvexHull(points)
    return hull


# find the barycenter of the convex hull
def getBarycenter(hull):
    pass


# estimate stroke density
def getStrokeDensity():
    pass


def main():
    # read in image
    img = cv2.imread("./data/sample-input.png", cv2.IMREAD_COLOR)
    num_rows, num_cols, num_channels = img.shape
    # turn image into vector of points
    points = []
    for i in range(num_rows):
        for j in range(num_cols):
            points.append(img[i,j].tolist())
    # find hull
    hull = getHull(points)
    # find barycenter from hull
    barycenter = getBarycenter(hull)


if __name__ == "__main__":
    main()
