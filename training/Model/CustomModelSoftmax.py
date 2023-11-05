from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch.nn as nn
import torch


class CustomModelSoftmax(nn.Module):
    def __init__(self, checkpoint):
        super(CustomModelSoftmax, self).__init__()
        self.model = AutoModel.from_config(
            AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768 * 4, 6)
        self.regressor = nn.Linear(768 * 4, 30)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        outputs = torch.cat((outputs[2][-1][:, 0, ...],
                             outputs[2][-2][:, 0, ...],
                             outputs[2][-3][:, 0, ...],
                             outputs[2][-4][:, 0, ...]), -1)
        outputs = self.dropout(outputs)
        outputs_classifier = self.classifier(outputs)
        outputs_regressor = self.regressor(outputs)
        outputs_classifier = nn.Sigmoid()(outputs_classifier)
        outputs_regressor = outputs_regressor.reshape(-1, 6, 5)
        return outputs_classifier, outputs_regressor