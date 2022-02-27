import datetime
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress import ProgressBarBase
from loguru import logger
import torch
import torch.nn as nn
from typing import Optional
from yolox.utils import get_world_size, get_rank, gpu_mem_usage, MeterBuffer
from pytorch_lightning.trainer.supporters import CombinedLoader
from yolox.data import DataLoader as YOLOXDataLoader
from yolox.evaluators import COCOEvaluator
import random
import torch.distributed as dist


class LogPredictionsCallback(Callback):

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pass


class ConfigureSchedulerCallback(Callback):
    # In pytorch Lightning datamodule and model are separated, however the scheduler in YOLOX require the
    # length of dataloader which violated the concept of pl. Therefore we use callback to bridge the gap between
    # model and datamodule, and the bridge is trainer which integrate two parts.
    def __init__(self, args):
        super(ConfigureSchedulerCallback, self).__init__()
        self.args = args

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info("Configure model's scheduler")
        max_iter = len(trainer.train_dataloader)
        scheduler = self.get_lr_scheduler(self.args, max_iter)
        pl_module.scheduler = scheduler

    @staticmethod
    def get_lr_scheduler(args, max_iter):
        from yolox.utils import LRScheduler
        lr = args.basic_lr_per_img * args.batch_size
        scheduler = LRScheduler(
            args.scheduler,
            lr,
            max_iter,
            args.max_epochs,
            warmup_epochs=args.warmup_epochs,
            warmup_lr_start=args.warmup_lr,
            no_aug_epochs=args.no_aug_epochs,
            min_lr_ratio=args.min_lr_ratio,
        )
        return scheduler


class NoAugCallback(Callback):
    def __init__(self, no_aug_epochs, save_path):
        super(NoAugCallback, self).__init__()
        self.no_aug_epochs = no_aug_epochs
        self.save_path = save_path / "last_mosaic_epoch.ckpt"

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch + 1 == trainer.max_epochs - self.no_aug_epochs or self.no_aug_epochs == -1:
            logger.info("--->Turn off mosaic augmentation, add L1 loss at the same time")
            train_dataloader = trainer.train_dataloader
            if isinstance(train_dataloader, CombinedLoader):
                train_dataloader.loaders.close_mosaic()
            elif isinstance(train_dataloader, YOLOXDataLoader):
                train_dataloader.close_mosaic()
            else:
                raise TypeError
            pl_module.use_l1 = True
            trainer.val_check_interval = 1
        else:
            trainer.save_checkpoint(self.save_path)


class ResizeInput(Callback):

    def __init__(self, multiscale_range=5, resize_interval=10, random_size=None):
        super(ResizeInput, self).__init__()
        self.multiscale_range = multiscale_range
        self.resize_interval = resize_interval
        self.random_size = random_size

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int,
                           **kwargs) -> None:
        if (pl_module.global_step + 1) % 10 == 0:
            tensor = torch.LongTensor(2).cuda()
            rank = get_rank()
            input_size = pl_module.input_size
            if rank == 0:
                size_factor = input_size[1] * 1.0 / input_size[0]
                if self.random_size is None:
                    min_size = int(input_size[0] / 32) - self.multiscale_range
                    max_size = int(input_size[0] / 32) + self.multiscale_range
                    self.random_size = (min_size, max_size)
                size = random.randint(*self.random_size)
                size = (int(32 * size), 32 * int(size * size_factor))
                tensor[0] = size[0]
                tensor[1] = size[1]

            if get_world_size() > 1:
                dist.barrier()
                dist.broadcast(tensor, 0)
            input_size = (tensor[0].item(), tensor[1].item())
            pl_module.input_size = input_size


class LogMetric(Callback):

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not hasattr(pl_module, "meter"):
            meter = MeterBuffer(window_size=trainer.log_every_n_steps)
            setattr(pl_module, "meter", meter)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int,
                           **kwargs) -> None:
        iter_end_time = time.time()
        pl_module.meter.update(
            iter_time=iter_end_time - outputs['iter_start_time'],
            data_time=outputs["data_end_time"] - outputs['iter_start_time'],
            loss=outputs['loss'],
            iou_loss=outputs["iou_loss"],
            conf_loss=outputs["conf_loss"],
            cls_loss=outputs['cls_loss'],
            l1_loss=outputs['l1_loss'],
            num_fg=outputs['num_fg']
        )
        max_iter = len(trainer.train_dataloader)
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            # TODO check ETA logic
            left_iters = max_iter * trainer.max_epochs - (trainer.global_step + 1)
            eta_seconds = pl_module.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                trainer.current_epoch + 1, trainer.max_epochs, batch_idx + 1, max_iter
            )
            loss_meter = pl_module.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = pl_module.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                )
                + (", size: {:d}, {}".format(pl_module.input_size[0], eta_str))
            )
            pl_module.meter.clear_meters()


class GetEvaluation(Callback):

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        val_dataloader = trainer.val_dataloaders[0]
        if not hasattr(pl_module, "evaluator"):
            evaluator = COCOEvaluator(val_dataloader,
                                      img_size=pl_module.test_size,
                                      confthre=pl_module.test_conf,
                                      nmsthre=pl_module.nmsthre,
                                      num_classes=pl_module.num_classes,
                                      testdev=pl_module.testdev)
            setattr(pl_module, "evaluator", evaluator)
