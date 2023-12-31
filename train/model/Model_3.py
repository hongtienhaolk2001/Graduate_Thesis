import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class CustomModel(nn.Module):
    def __init__(self, checkpoint):
        super(CustomModel, self).__init__()
        # Load pretrained model with attention and hidden states output
        self.model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint,
                                                                      output_attentions=True,
                                                                      output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(1024 * 4, 5)
        self.regressor = nn.Linear(1024 * 4, 20)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Concatenate the last four hidden states
        outputs = torch.cat((outputs[2][-1][:, 0, ...],
                             outputs[2][-2][:, 0, ...],
                             outputs[2][-3][:, 0, ...],
                             outputs[2][-4][:, 0, ...]), -1)
        outputs = self.dropout(outputs)
        outputs_classifier = self.classifier(outputs)
        outputs_regressor = self.regressor(outputs)
        outputs_classifier = nn.Sigmoid()(outputs_classifier)
        outputs_regressor = outputs_regressor.reshape(-1, 5, 4)
        return outputs_classifier, outputs_regressor

