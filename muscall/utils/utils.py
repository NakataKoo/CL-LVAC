import os
import random
from io import open
import json
import numpy as np
from omegaconf import OmegaConf

import torch
import torch.functional as F

# YAML形式の設定ファイルを読み込み、環境設定 (env.base_dir) が未設定であれば、ルートディレクトリを設定してから、その設定オブジェクトを返す
def load_conf(path_to_yaml):
    """Wrapper for configuration file loading through OmegaConf."""
    conf = OmegaConf.load(path_to_yaml)
    if "env" in conf.keys() and conf.env.base_dir is None:
        OmegaConf.update(conf, "env.base_dir", get_root_dir())
    return conf


# 複数の設定ファイル（ベース設定、データセット設定、モデル設定）を読み込み、それらを1つの設定オブジェクトに統合
def merge_conf(base_conf_path, dataset_conf_path, model_conf_path):
    """Wrapper for to merge multiple config files through OmegaConf."""
    base_conf = load_conf(base_conf_path)
    dataset_conf = load_conf(dataset_conf_path)
    model_conf = load_conf(model_conf_path)

    conf = OmegaConf.merge(base_conf, dataset_conf, model_conf)
    return conf


def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# コマンドライン引数で指定されたパラメータを設定オブジェクトに反映させ、設定を動的に更新
def update_conf_with_cli_params(params, config):
    params_dict = vars(params)
    for param in params_dict:
        if params_dict[param] is not None:
            # TODO automatically find mapping for arbitrary depth
            if param in config.keys():
                new_param = params_dict[param]
                if isinstance(new_param, str) and new_param.lower() in [
                    "true",
                    "false",
                ]:
                    new_param = new_param.lower() == "true"
                OmegaConf.update(config, "{}".format(param), new_param)
            else:
                for top_level_key in config.keys():
                    if isinstance(config[top_level_key], dict):
                        if param in list(config[top_level_key].keys()):
                            new_param = params_dict[param]
                            if isinstance(new_param, str) and new_param.lower() in [
                                "true",
                                "false",
                            ]:
                                new_param = new_param.lower() == "true"
                            OmegaConf.update(
                                config, "{}.{}".format(top_level_key, param), new_param
                            )


def save_json(output_path, content):
    with open(output_path, "w") as outfile:
        json.dump(content, outfile)

# 現在のスクリプトの位置から2つ上のディレクトリの絶対パスを取得
def get_root_dir():
    # TODO: below should be run only once, then saved
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root, "../.."))
    return root


def normalize(x, p, dim):
    norm = F.norm(x, p=p, dim=dim)
    return x / norm


def scale(x, axis=0):
    mean = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    scale = std[std == 0.0] = 1.0
    x -= mean
    x /= scale
    return np.array(x, dtype=np.float32)
