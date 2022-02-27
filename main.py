import sys
from argparse import ArgumentParser
from yolox.models import YOLOX, YOLOXHead, YOLOPAFPN, YOLOXDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import torch
from yolox.models.callbacks import *
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(ROOT)


def make_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--min_lr_ratio",
        default=0.05,
        type=float
    )
    parser.add_argument(
        "--basic_lr_per_img",
        default=0.01 / 64.0,
        type=float
    )
    parser.add_argument(
        "--scheduler",
        default="yoloxwarmcos",
        type=str
    )
    parser.add_argument(
        "--resize_interval",
        default=10,
        type=int
    )
    parser.add_argument(
        "--save_history_ckpt",
        action="store_true",
    )
    parser.add_argument(
        "--exp_name",
        default="yolox",
        type=str
    )
    # configuration shared by head and backbone
    parser.add_argument(
        "--width",
        default=1.0,
        type=float,
        help="width of the network",
    )
    parser.add_argument(
        "--act",
        default="silu",
        type=str,
        help="activation function",
    )
    parser.add_argument(
        "--depthwise",
        action='store_true',
        help="whether to use depthwise convolution",
    )
    parser.add_argument(
        "--nmsthre",
        default=0.65,
        type=float
    )
    parser.add_argument(
        "--log_dir",
        default="/log",
        type=str,
        help="path to save log and local files(checkpoints, profiler traces, etc)"
    )
    parser = YOLOXDataModule.add_argparse_args(parser, use_argument_group=True)
    parser = pl.Trainer.add_argparse_args(parser, use_argument_group=True)
    parser = YOLOXHead.add_model_specific_args(parser)
    parser = YOLOPAFPN.add_model_specific_args(parser)
    parser = YOLOX.add_model_specific_args(parser)
    return parser


def main(args):
    # configure arguments with type like list/tuple etc
    input_size = (320, 320)
    test_size = (320, 320)
    mosaic_scale = (0.5, 1.5)
    mixup_scale = (0.5, 1.5)
    random_size = None
    strides = [8, 16, 32]
    in_channels = [256, 512, 1024]
    in_features = ["dark3", "dark4", "dark5"]
    log_dir = ROOT/"log"
    args.no_aug = args.no_aug_epochs == -1

    backbone = YOLOPAFPN(depth=args.depth,
                         width=args.width,
                         in_features=in_features,
                         in_channels=in_channels,
                         depthwise=args.depthwise,
                         act=args.act)

    head = YOLOXHead(num_classes=args.num_classes,
                     act=args.act,
                     depthwise=args.depthwise,
                     width=args.width,
                     strides=strides,
                     in_channels=in_channels)

    model = YOLOX(args,
                  input_size,
                  test_size,
                  backbone=backbone,
                  head=head)

    data_module = YOLOXDataModule.from_argparse_args(args)
    data_module.get_tuple_param(input_size=input_size,
                                test_size=test_size,
                                mosaic_scale=mosaic_scale,
                                mixup_scale=mixup_scale)

    wandb_logger = WandbLogger(project="YOLOX",
                               save_dir=log_dir,
                               offline=True)

    checkpoint_callback = ModelCheckpoint(save_last=True,
                                          auto_insert_metric_name=True,
                                          save_on_train_epoch_end=True,
                                          monitor="ap50_95")
    lr_monitor = LearningRateMonitor()
    optimizer_callback = ConfigureSchedulerCallback(args)

    noaug_callback = NoAugCallback(args.no_aug_epochs, log_dir)

    resize_callback = ResizeInput(multiscale_range=args.multiscale_range,
                                  resize_interval=args.resize_interval,
                                  random_size=random_size)
    log_metric_callback = LogMetric()
    build_evaluator_callback = GetEvaluation()

    trainer = pl.Trainer(logger=wandb_logger,
                         gpus=1,
                         enable_progress_bar=False,
                         log_every_n_steps=args.log_every_n_steps,
                         val_check_interval=args.val_check_interval,
                         max_epochs=args.max_epochs,
                         callbacks=[checkpoint_callback,
                                    lr_monitor,
                                    optimizer_callback,
                                    noaug_callback,
                                    resize_callback,
                                    log_metric_callback,
                                    build_evaluator_callback
                                    ])
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)
