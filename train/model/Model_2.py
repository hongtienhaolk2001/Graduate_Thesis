import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class CustomModelSoftmax(nn.Module):
    def __init__(self, checkpoint):
        super(CustomModelSoftmax, self).__init__()
        # Load pretrained model with attention and hidden states output
        self.model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint,
                                                                      output_attentions=True,
                                                                      output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        # Classifier for classification task
        self.classifier = nn.Linear(in_features=768 * 4, out_features=6)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Concatenate the last four hidden states
        outputs = torch.cat((outputs[2][-1][:, 0, ...],
                             outputs[2][-2][:, 0, ...],
                             outputs[2][-3][:, 0, ...],
                             outputs[2][-4][:, 0, ...]), -1)
        outputs = self.dropout(outputs)
        outputs_classifier = self.classifier(outputs)
        return outputs_classifier
