import numpy as np



def adjust_box_anns(bbox, scale, padw, padh, w_max, h_max):
    bbox[:, ::2] = bbox[:, ::2] * scale + padw
    bbox[:, 1::2] = bbox[:, 1::2] * scale + padh

    np.clip(bbox[:, ::2], padw, w_max, out=bbox[:, ::2])
    np.clip(bbox[:, 1::2], padh, h_max, out=bbox[:, 1::2])

    return bbox