import os
import torch
import time
from omegaconf import OmegaConf


# 実験のログを管理し、実験設定やモデルのチェックポイントを保存
class Logger:
    def __init__(self, config):
        self.config = config

        self.init_pretrain_logger()
        self.init_training_log()

    def init_pretrain_logger(self):
        if self.config.env.experiment_id is None:
            self.experiment_id = self.get_timestamp() # 現在時刻を実験idとする
            OmegaConf.update(self.config, "env.experiment_id",
                             self.experiment_id)
        else:
            self.experiment_id = self.config.env.experiment_id
        
        self.experiment_dir = os.path.join(self.config.env.experiments_dir,
                                           self.experiment_id) # 実験結果を保存するディレクトリ
        self.checkpoint_path = os.path.join(
            self.experiment_dir, 'checkpoint.pth.tar') # チェックポイントの保存先

    def init_training_log(self):
        self.log_filename = os.path.join(self.experiment_dir, "train_log.tsv")
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            log_file = open(self.log_filename, 'a') # train_log.tsv ファイルを追記モード ('a') で開く
            log_file.write(
                'Epoch\ttrain_loss\tval_loss\tmetric\tepoch_time\tlearing_rate\ttime_stamp\n'
            ) # ログファイルのヘッダーを書き込む（ヘッダーには、エポック、トレーニング損失、検証損失、メトリック、エポック時間、学習率、タイムスタンプが含まれる）
            log_file.close()

    def save_config(self):
        # Save json with experiment settings
        config_path = os.path.join(self.experiment_dir, 'config.yaml')
        if not os.path.exists(config_path):
            OmegaConf.save(self.config, config_path)

    def get_timestamp(self):
        return str(time.strftime('%Y-%m-%d-%H_%M_%S', time.gmtime()))

    def write(self, text):
        print(text)

    # トレーニングの進行状況をログファイルtrain_log.tsvに記録
    def update_training_log(self, epoch, train_loss, val_loss, epoch_time,
                            learning_rate, metric=0):
        time_stamp = str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))
        self.write(
            'Epoch %d, train loss %g, val loss %g, metric %g, epoch-time %gs, lr %g, time-stamp %s'
            %
            (epoch, train_loss, val_loss, metric, epoch_time, learning_rate, time_stamp))

        log_file = open(self.log_filename, 'a')
        log_file.write('%d\t%g\t%g\t%g\t%gs\t%g\t%s\n' %
                       (epoch, train_loss, val_loss, metric, epoch_time, learning_rate,
                        time_stamp))
        log_file.close()

    # モデルの状態をディスクに保存し、必要に応じて最良のモデルも別途保存
    def save_checkpoint(self, state, is_best=False):
        torch.save(state, self.checkpoint_path)
        if is_best:
            self.write("Saving best model so far")
            best_model_path = os.path.join(self.experiment_dir,
                                           'best_model.pth.tar')
            torch.save(state, best_model_path)
