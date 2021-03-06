# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from finetune_param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_CLF_LENGTH = 40


class ClassifierModel(nn.Module):
    def __init__(self, num_answers, model_type='full'):
        super().__init__()
        self.model_type = model_type
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_CLF_LENGTH,
            model_type=args.model_type)
        
        hid_dim = self.lxrt_encoder.dim

        if num_answers == 2:
            output_dim = 1
        else:
            output_dim = num_answers

        if self.model_type != 'concat':
            self.logit_fc = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(hid_dim, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                nn.Dropout(args.dropout),
                nn.Linear(hid_dim * 2, output_dim)
            )
        else:
            linear = nn.Linear(hid_dim, output_dim)
            self.logit_fc = nn.Sequential(
                nn.Dropout(args.dropout),
                linear,
            )
            
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit


