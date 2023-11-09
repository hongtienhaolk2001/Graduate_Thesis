from transformers import AutoModel, AutoConfig
from CustomTransformer import *


class CustomModelMultiHead(nn.Module):
    def __init__(self, checkpoint):
        super(CustomModelMultiHead, self).__init__()
        self.model = AutoModel.from_config(
            AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 6)
        self.regressor = nn.Linear(768, 30)
        self.encoderlayer = EncoderLayer(768, 64, 64, 12, 3096)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        outputs = self.dropout(outputs[0])
        outputs = self.encoderlayer(outputs, attention_mask.type(torch.bool))  # N, S, D
        outputs = outputs[:, 0, :]
        outputs_classifier = self.classifier(outputs)
        outputs_regressor = self.regressor(outputs)
        outputs_regressor = outputs_regressor.reshape(-1, 6, 5)
        outputs_classifier = nn.Sigmoid()(outputs_classifier)
        return outputs_classifier, outputs_regressor