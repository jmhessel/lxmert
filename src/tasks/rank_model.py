# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
from lxrt.modeling import GeLU, BertLayerNorm
from lxrt.entry import LXRTEncoder
from finetune_param import args

# Max length including <bos> and <eos>
MAX_RANK_LENGTH = 40


class RankModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_RANK_LENGTH
        )
        self.hid_dim = hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, 1)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)


    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, 1) The score of this pair
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit
