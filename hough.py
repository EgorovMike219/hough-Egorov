from __future__ import print_function
from sys import argv
import cv2
import numpy as np
from math import pi, sin, cos, sqrt
from skimage.feature import peak_local_max


def gradient_img(img):
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)

    return magnitude


def hough_transform(img, theta, rho):
    h, w = img.shape
    max_theta = int(2 * pi / theta + 0.5)  # угл от 0 до пи
    max_rho = int(sqrt(h*h + w*w) / rho + 0.5)  # расстояние от 0 до max_rho
    hough_img = np.zeros((max_rho, max_theta))

    theta = np.arange(max_theta)
    for y in range(h):
        for x in range(w):
            r = np.array(np.rint(x*np.cos(theta / 180 * pi) + y*np.sin(theta / 180 * pi)), dtype=np.int64)
            hough_img[r, theta] += img[y, x]

    # theta = np.arange(max_theta)
    # y = np.arange(h)
    # x = np.arange(w)
    # xs, ys, thetas = np.meshgrid(x, y, theta)
    # rs = np.array(np.rint(xs*np.cos(thetas / 180 * pi) + ys*np.sin(thetas / 180 * pi)), dtype=np.int64)
    # hough_img[rs, thetas] += img[ys, xs]
    
    hough_img = (255 * (hough_img.astype(float) / np.max(hough_img))).astype(np.uint8)
    coordinates = np.array(peak_local_max(hough_img, min_distance=10))
    return hough_img, coordinates[:, 1], coordinates[:, 0]


def get_lines(ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta):
    coord = [(ht_map[y, x], x / 180 * pi, y) for y, x in zip(rhos, thetas)]
    coord.sort(key=lambda x: x[0], reverse=True)
    coord = coord[:min(n_lines, len(coord))]
    result = []
    for conf_x_y in coord:
        x, y = conf_x_y[1], conf_x_y[2]
        if y > min_delta_rho and x > min_delta_theta:
            result.append([y, x])
    return result

if __name__ == '__main__':
    assert len(argv) == 9
    src_path, dst_ht_path, dst_lines_path, theta, rho,\
        n_lines, min_delta_rho, min_delta_theta = argv[1:]

    theta = float(theta)
    rho = float(rho)
    n_lines = int(n_lines)
    min_delta_rho = float(min_delta_rho)
    min_delta_theta = float(min_delta_theta)

    assert theta > 0.0
    assert rho > 0.0
    assert n_lines > 0
    assert min_delta_rho > 0.0
    assert min_delta_theta > 0.0

    image = cv2.imread(src_path, 0)
    assert image is not None
    h, w = image.shape

    image = image.astype(float)
    gradient = gradient_img(image)

    ht_map, thetas, rhos = hough_transform(gradient,
                                           theta, rho)
    cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(ht_map, n_lines, thetas, rhos,
                      min_delta_rho, min_delta_theta)
    image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2RGB)
    with open(dst_lines_path, 'w') as fout:
        for line in lines:
            fout.write('%0.3f, %0.3f\n' % (line[0], line[1]))
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),4)
    
    cv2.imwrite('houghlines.jpg',image)
