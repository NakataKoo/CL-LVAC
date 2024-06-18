import numpy as np

import torch
from torch import nn
from transformers import CLIPTextModel

from muscall.modules.textual_heads import TextTransformer
from muscall.modules.audio_ssl import SimCLRAudio
from muscall.modules.audio_backbones import ModifiedResNet

# クロスエントロピー誤差
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)

# 文の類似度に基づいて重みを付けた対照損失
def weighted_loss(logits, sentence_sim, k=0.01):
    batch_size = logits.size(0)
    mask = 1 - torch.eye(batch_size).to(device=logits.device)

    sentence_sim = (sentence_sim * mask).mean(-1)

    normed_sim = sentence_sim / sentence_sim.sum()
    weight = torch.exp(normed_sim / k)

    labels = torch.arange(len(logits), device=logits.device)
    loss = weight * nn.functional.cross_entropy(logits, labels, reduction="none")
    loss = loss.sum() / weight.sum()

    return loss

# 誤差関数
def clip_loss(similarity: torch.Tensor, sentence_sim=None, type_loss="clip") -> torch.Tensor:
    if sentence_sim is not None and type_loss == "weighted_clip":
        text_loss = weighted_loss(similarity, sentence_sim)
        audio_loss = weighted_loss(similarity.T, sentence_sim)
    else:
        text_loss = contrastive_loss(similarity)
        audio_loss = contrastive_loss(similarity.T)
    return (text_loss + audio_loss) / 2.0


class MusCALL(nn.Module):
    def __init__(self, config):
        super().__init__()
        audio_config = config.audio # 音声エンコーダの設定
        text_config = config.text # テキストエンコーダの設定

        projection_dim = config.projection_dim
        audio_dim = audio_config.hidden_size
        text_dim = text_config.hidden_size

        self.do_audio_ssl = audio_config.ssl.do_ssl
        self.audio_ssl_loss_weight = (
            audio_config.ssl.ssl_loss_weight if self.do_audio_ssl else 0
        )

        self.type_loss = config.loss

        self.temperature = config.temperature

        if config.audio.model == "ModifiedResNet":
            self.audio_backbone = ModifiedResNet(audio_config)
        if config.text.model == "TextTransformer":
            self.textual_head = TextTransformer(text_config)
        elif config.text.model == "CLIPTextModel":
            pretrained_model = config.text.pretrained
            self.textual_head = CLIPTextModel.from_pretrained(pretrained_model)

        self.audio_projection = nn.Linear(audio_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, projection_dim, bias=False)

        if self.temperature is None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.do_audio_ssl:
            print("Running audio SSL")
            self.audio_ssl = SimCLRAudio(
                encoder=self.audio_backbone,
                audio_config=audio_config,
            )

    def encode_audio(self, audio):
        audio_features = self.audio_backbone(audio)
        audio_features = self.audio_projection(audio_features)
        return audio_features

    def encode_text(self, text, text_mask):
        if isinstance(self.textual_head, TextTransformer):
            text_features = self.textual_head(text, text_mask)
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            pooled_outout = text_features[
                torch.arange(text_features.shape[0]), text.argmax(dim=-1)
            ]
        elif isinstance(self.textual_head, CLIPTextModel):
            outputs = self.textual_head(text, text_mask)
            pooled_outout = outputs.pooler_output

        text_features = self.text_projection(pooled_outout)
        return text_features

    # 音声とテキストの特徴をエンコードし、対照学習のための損失を計算
    def forward(
        self,
        audio, # 拡張後の音声データ（バッチ）
        text, # テキストデータ（バッチ）
        original_audio=None, # 元音声データ
        sentence_sim=None, # 文の類似度(オプション)
        text_mask=None, #テキストのマスク(オプション)
        return_loss=True, # 損失を計算して返すかどうかを指定するフラグ
    ):
        """
        以下のような__call__メソッドがオーバーロードされており、
        モデルインスタンス(self.model(x, y, ...))を関数のように呼び出すと、
        自動的にforwardメソッドを実行
        
        def __call__(self, *input, **kwargs):
            return self.forward(*input, **kwargs)

        従って、以下の2つの呼び出しは等価

        # 直接 forward メソッドを呼び出す
        loss = self.model.forward()

        # 間接的に __call__ メソッドを介して forward メソッドを呼び出す
        loss = self.model()

        """
        if return_loss:
            """
            ・音声自己教師付き学習（SSL）の損失を計算
            ・元の音声データと変換された音声データを比較し、同一性を学習
            """
            audio_ssl_loss = (
                self.audio_ssl(audio, original_audio) if self.do_audio_ssl else 0
            )

        # 音声とテキストの特徴をそれぞれエンコード
        audio_features = self.encode_audio(audio)
        text_features = self.encode_text(text, text_mask)

        # normalise features（各特徴ベクトルをそのノルムで割ることで、単位ベクトルに変換）
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # ロジットの計算
        # ロジットスケール（温度パラメータ）を計算。温度が未設定の場合、学習されたlogit_scaleを使用
        if self.temperature is None:
            logit_scale = self.logit_scale.exp()
        # 音声とテキストの特徴ベクトルの内積を計算してロジットを得る
        else:
            logit_scale = 1.0 / self.temperature
        logits_per_audio = logit_scale * audio_features @ text_features.t()
        logits_per_text = logits_per_audio.t()

        # マルチモーダル損失を計算
        if return_loss:
            multimodal_loss = clip_loss(
                logits_per_text, sentence_sim, type_loss=self.type_loss
            )

            clip_loss_weight = 1 - self.audio_ssl_loss_weight
            loss = (multimodal_loss * clip_loss_weight) + (
                audio_ssl_loss * self.audio_ssl_loss_weight
            )

            return loss
        else:
            return logits_per_audio, logits_per_text

    @classmethod
    def config_path(cls):
        return "configs/models/muscall.yaml"
