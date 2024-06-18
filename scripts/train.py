import argparse
import os
from omegaconf import OmegaConf

from muscall.utils.logger import Logger
from muscall.utils.utils import (
    load_conf,
    merge_conf,
    get_root_dir,
    update_conf_with_cli_params,
)
from muscall.models.muscall import MusCALL
from muscall.trainers.muscall_trainer import MusCALLTrainer
from muscall.datasets.audiocaption import AudioCaptionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MusCALL model")

    parser.add_argument(
        "--experiment_id",
        type=str,
        help="experiment id under which checkpoint was saved",
        default=None,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="path to base config file",
        default=os.path.join(get_root_dir(), "configs", "training.yaml"), # デフォルトではtraining.yamlを読み込む
    )
    parser.add_argument(
        "--dataset", type=str, help="name of the dataset", default="audiocaption"
    )
    parser.add_argument("--device_num", type=str, default="0")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    params = parse_args()

    if params.experiment_id is None:
        # 1. Load config (base + dataset + model)
        base_conf = load_conf(params.config_path) # デフォルトではtraining.yamlを読み込む

        if params.dataset == "audiocaption":
            dataset_conf_path = os.path.join(base_conf.env.base_dir, AudioCaptionDataset.config_path()) # "configs/datasets/audiocaption.yaml"のパスを取得
        else:
            raise ValueError("{} dataset not supported".format(params.dataset))

        model_conf_path = os.path.join(base_conf.env.base_dir, MusCALL.config_path()) # "configs/models/muscall.yaml"のパスを取得

        config = merge_conf(params.config_path, dataset_conf_path, model_conf_path) # 「config＝training設定＋データセット設定＋モデル設定」(config.yamlが学習設定ファイルとして保存される)
        update_conf_with_cli_params(params, config) # 引数の値をconfigに反映
    else:
        config = OmegaConf.load(
            "./save/experiments/{}/config.yaml".format(params.experiment_id)
        )

    logger = Logger(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_num

    trainer = MusCALLTrainer(config, logger)
    print("# of trainable parameters:", trainer.count_parameters()) # 学習パラメータ数の表示

    trainer.train() # モデル学習開始
