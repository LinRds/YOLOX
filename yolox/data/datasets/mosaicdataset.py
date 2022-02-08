import random
import cv2
import numpy as np
from utils import adjust_box_anns

from datasets_wrapper import Dataset
from ..data_augment import random_affine


def get_mosaic_coordinate(xc, yc, patch_w, patch_h, mosaic_w, mosaic_h, index):
    """
    :param xc: x center of mosaic image
    :param yc: y center of mosaic image
    :param patch_w: width of the patch
    :param patch_h: height of the patch
    :param mosaic_w: width of the mosaic image
    :param mosaic_h: height of the mosaic image
    :param index: 0, 1, 2, 3 represents top left, top right, bottom left, bottom right respectively
    """
    index = index % 4
    x1, y1, x2, y2 = 0, 0, 0, 0
    patch_coordinates = ()
    if index == 0:
        x1, y1, x2, y2 = max(0, xc - patch_w), max(0, yc - patch_h), xc, yc
        patch_coordinates = max(0, patch_w - xc), max(0, patch_h - yc), patch_w, patch_h
    elif index == 1:
        x1, y1, x2, y2 = xc, max(yc - patch_h, 0), mosaic_w, yc
        patch_coordinates = 0, patch_h - (y2 - y1), min(x2 - x1, patch_w), patch_h
    elif index == 2:
        x1, y1, x2, y2 = max(xc - patch_w, 0), yc, xc, min(mosaic_h - patch_h, mosaic_h)
        patch_coordinates = patch_w - (x2 - x1), 0, patch_w, min(y2 - y1, patch_h)
    elif index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + patch_w, mosaic_w), min(yc + patch_h, mosaic_h)
        patch_coordinates = 0, 0, min(x2 - x1, patch_w), min(y2 - y1, patch_h)
    return (x1, y1, x2, y2), patch_coordinates

class MosaicDataset(Dataset):
    def __init__(self, dataset, image_size, mosaic=True,
                 preproc=None, degrees=10.0, translate=0.1,
                 mosaic_scale=(0.5, 1.5), mixup_scale=(0.5, 1.5), shear=2.0,
                 mixup=True, mosaic_prob=1.0, mixup_prob=1.0, *args):
        super(MosaicDataset, self).__init__(image_size)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.mosaic_scale = mosaic_scale
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.shear = shear

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            idx = [index]
            idx = idx + [random.choice(range(len(self._dataset))) for _ in range(3)]
            input_h, input_w = self._dataset.input_dim

            mosaic_img = np.full((input_h, input_w), 114, dtype=np.uint8)
            xc = int(np.random.uniform(0.25, 0.75) * input_w * 2)
            yc = int(np.random.uniform(0.25, 0.75) * input_h * 2)
            for i, item in enumerate(idx):
                img, _labels, _, img_id = self._dataset.pull_item(item)
                h, w, c = img.shape
                scale = min(input_h / h, input_w / w)
                img = cv2.resize(img, (int(h * scale), int(w * scale)), interpolation=cv2.INTER_LINEAR)
                if i == 0:
                    mosaic_img = mosaic_img[:, :, np.newaxis].repeat(c, axis=2)
                (x1, y1, x2, y2), (patch_x1, patch_x2, patch_y1, patch_y2) = get_mosaic_coordinate(xc, yc, w, h, input_w * 2, input_h * 2, i)
                mosaic_img[y1:y2, x1:x2] = img[patch_y1:patch_y2, patch_x1:patch_x2]

                shift_h, shift_w = (y1 - patch_y1), (x1 - patch_x1)

                labels = _labels.copy()
                if len(labels) > 0:
                    labels[:, 0] = _labels[:, 0] * scale + shift_w
                    labels[:, 1] = _labels[:, 1] * scale + shift_h
                    labels[:, 2] = _labels[:, 2] * scale + shift_w
                    labels[:, 3] = _labels[:, 3] * scale + shift_h
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, axis=0)
                np.clip(mosaic_labels[:, ::2], 0, input_w * 2, out=mosaic_labels[:, ::2])
                np.clip(mosaic_labels[:, 1::2], 0, input_h * 2, out=mosaic_labels[:, 1::2])

            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_h, input_w),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.mosaic_scale,
                shear=self.shear
            )

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (
                    self.enable_mixup
                    and not len(mosaic_labels) == 0
                    and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            return mix_img, padded_labels, img_info, img_id

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id = self._dataset.pull_item(index)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*[0.5, 1.5])
        FLIP = random.uniform(0, 1) > 0.1
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)

        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]

        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]
        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )

        if FLIP:
            cp_bboxes_origin_np[:, 0] = (
                    origin_w - cp_bboxes_origin_np[:, 0] - cp_bboxes_origin_np[:, 2]
            )

        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels






