import math
import random

import cv2
import numpy as np


def random_affine(images,
                  labels,
                  target_size,
                  degrees,
                  translate,
                  scales,
                  shear):
    M, scales = get_affine_matrix(target_size, degrees, translate, scales, shear)
    if len(labels) > 0:
        labels = apply_affine_to_bboxes(labels, target_size, M)

    images = cv2.warpAffine(images, M, target_size, borderValue=(114, 114, 114))

    return images, labels



def get_aug_param(endpoints, center=0):
    """
    if `endpoints` is a scalar, it is treated as half the length of an interval whose center is defined by param `center`;
    otherwise, it should be an iterable object of length 2, the left and right endpoints are defined explicitly.
    :param endpoints:
    :param center:
    :return:
    """
    if isinstance(endpoints, float):
        param = random.uniform(center - endpoints, center + endpoints)
    elif len(endpoints) == 2:
        param = random.uniform(endpoints[0], endpoints[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
                          or single float values. Got {}".format(
                endpoints
            )
        )
    return param

def get_affine_matrix(
        target_size,
        degrees=10,
        translate=0.1,
        scales=0.1,
        shear=10
):
    tw, th = target_size
    angle = get_aug_param(degrees)
    scales = get_aug_param(scales, center=1.0)

    if scales <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scales)

    M = np.ones([2, 3])

    # shear
    shear_x = math.tan(get_aug_param(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_param(shear) * math.pi / 180)
    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # translation
    translate_x = get_aug_param(translate) * tw
    translate_y = get_aug_param(translate) * th
    M[:2, -1] = translate_x, translate_y

    return M, scales


def apply_affine_to_bboxes(bbox, target_size, affine_matrix):
    tw, th = target_size

    corner_points = bbox[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(-1, 2)
    corner_points = cv2.transform(corner_points[np.newaxis, ...], affine_matrix).squeeze()

    # clip
    np.clip(corner_points[:, 0], 0., tw, out=corner_points[:, 0])
    np.clip(corner_points[:, 1], 0., th, out=corner_points[:, 1])

    corner_points = corner_points.flatten().reshape(-1, 4 * 2)

    bbox[:, 0] = corner_points[:, ::2].min(axis=-1)
    bbox[:, 1] = corner_points[:, 1::2].min(axis=-1)
    bbox[:, 2] = corner_points[:, ::2].max(axis=-1)
    bbox[:, 3] = corner_points[:, 1::2].max(axis=-1)

    return bbox




