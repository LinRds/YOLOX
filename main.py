from pytorch_lightning.utilities.cli import LightningArgumentParser


def make_parser(lighting_class, config_path):
    parser = LightningArgumentParser(parser_mode="yaml")
    parser.add_lightning_class_args(lighting_class)