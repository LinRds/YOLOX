from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import pytorch_lightning as pl
from yolox.data import COCODataset, TrainTransform, YoloBatchSampler, DataLoader, \
    InfiniteSampler, MosaicDataset, worker_init_reset_seed, ValTransform
from yolox.utils import wait_for_the_master, get_local_rank, get_world_size
import os
from typing import Any, List, Dict, Tuple, Union
import torch.distributed as dist
import torch


class YOLOXDataModule(pl.LightningDataModule):
    """
    Data module class for YOLOX data sets.
    This class handles configurations for training on data. It is set
    up to process configurations independently of training modules.
    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """
    data_num_workers: int

    def __init__(
            self,
            batch_size: int = 16,
            no_aug_epochs: int = -1,
            cache_img: bool = False,
            data_num_workers: int = 4,
            multiscale_range: int = 5,
            train_ann: str = None,
            val_ann: str = None,
            test_ann: str = None,
            data_dir: str = None,
            degrees: float = 10,
            translate: float = 0.1,
            enable_mixup: bool = True,
            enable_mosaic: bool = True,
            mosaic_prob: float = 1.0,
            mixup_prob: float = 1.0,
            shear: float = 2.0,
            seed: int = 0,
            legacy: bool = False,
            max_labels: int = 50,
            flip_prob: float = 0.5,
            hsv_prob: float = 1.0
    ):
        """
        :param batch_size:
        :param no_aug_epochs:
        :param cache_img:
        :param data_num_workers:
        :param multiscale_range:
        :param train_ann: name of annotation file for training
        :param val_ann: name of annotation file for evaluation
        :param test_ann: name of annotation file for testing
        :param data_dir: dir of dataset images, if data_dir is None, this project will use `datasets` dir
        :param degrees: rotation angle range, for example, if set to 2, the true range is (-2, 2)
        :param translate: translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        :param enable_mixup: apply mixup aug or not
        :param enable_mosaic:
        :param mosaic_prob: prob of applying mosaic aug
        :param mixup_prob: prob of applying mixup aug
        :param shear: shear angle range, for example, if set to 2, the true range is (-2, 2)
        """
        super(YOLOXDataModule, self).__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.is_distributed = get_world_size() > 1
        self.no_aug = no_aug_epochs == -1
        self.cache_img = cache_img
        self.data_num_workers = data_num_workers
        self.multiscale = multiscale_range
        self.train_ann = train_ann
        self.val_ann = val_ann
        self.test_ann = test_ann
        self.degrees = degrees
        self.translate = translate
        self.enable_mosaic = enable_mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.shear = shear
        self.seed = seed
        if data_dir is None:
            data_dir = self.get_data_dir()
        self.data_dir = data_dir

        # transform
        self.train_transform = TrainTransform(max_labels=max_labels,
                                              flip_prob=flip_prob,
                                              hsv_prob=hsv_prob)
        self.val_transform = ValTransform(legacy=legacy)

    def get_tuple_param(self, input_size, test_size, mosaic_scale, mixup_scale):
        self.input_size = input_size
        self.test_size = test_size
        self.mosaic_scale = mosaic_scale
        self.mixup_scale = mixup_scale

    def setup(self, stage=None):
        local_rank = get_local_rank()
        if stage == "fit" or stage is None:
            with wait_for_the_master(local_rank):
                self.train_dataset = COCODataset(
                    data_dir=self.data_dir,
                    json_file=self.train_ann,
                    image_size=self.input_size,
                    preproc=self.train_transform
                )

            self.train_dataset = MosaicDataset(
                self.train_dataset,
                mosaic=not self.no_aug,
                image_size=self.input_size,
                preproc=self.train_transform,
                degrees=self.degrees,
                translate=self.translate,
                mosaic_scale=self.mosaic_scale,
                mixup_scale=self.mixup_scale,
                shear=self.shear,
                mixup=self.enable_mixup,
                mosaic_prob=self.mosaic_prob,
                mixup_prob=self.mixup_prob,
            )

            self.val_dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.val_ann,
                name="val2017",
                image_size=self.test_size,
                preproc=self.val_transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.test_ann,
                name="test2017",
                image_size=self.test_size,
                preproc=self.val_transform,
            )
        if stage == "predict" or stage is None:
            self.test_dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.test_ann,
                name="test2017",
                image_size=self.test_size,
                preproc=self.val_transform,
            )

    def train_dataloader(self):
        batch_size = self.batch_size
        if self.is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.train_dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not self.no_aug,
        )

        dataloader_kwargs = dict(num_workers=self.data_num_workers,
                                 pin_memory=True,
                                 batch_sampler=batch_sampler,
                                 worker_init_fn=worker_init_reset_seed)

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.

        train_loader = DataLoader(self.train_dataset, **dataloader_kwargs)
        return train_loader

    def val_dataloader(self):
        return self._shared_dataloader(stage="val")

    def test_dataloader(self):
        return self._shared_dataloader(stage="test")

    def predict_dataloader(self):
        return self._shared_dataloader(stage="test")

    def _shared_dataloader(self, stage="val"):
        batch_size = self.batch_size
        if self.is_distributed:
            batch_size = self.batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        dataset = self.val_dataset
        if stage == "test":
            dataset = self.test_dataset
        dataloader_kwargs = dict(num_workers=self.data_num_workers,
                                 pin_memory=True,
                                 sampler=sampler,
                                 batch_size=batch_size)
        data_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        return data_loader

    @staticmethod
    def get_data_dir():
        """
        get dataset dir of YOLOX. If environment variable named `YOLOX_DATADIR` is set,
        this function will return value of the environment variable. Otherwise, use data
        """
        yolox_datadir = os.getenv("YOLOX_DATADIR", None)
        if yolox_datadir is None:
            import yolox

            yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
            yolox_datadir = os.path.join(yolox_path, "datasets")
        return yolox_datadir
