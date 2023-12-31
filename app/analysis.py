import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from app.preprocessing import Preprocess


class Analysis(nn.Module):
    def __init__(self, checkpoint):
        super(Analysis, self).__init__()
        self.model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint,
                                                                      output_attentions=True,
                                                                      output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768 * 4, 5)
        self.regressor = nn.Linear(768 * 4, 20)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = torch.cat((outputs[2][-1][:, 0, ...],
                             outputs[2][-2][:, 0, ...],
                             outputs[2][-3][:, 0, ...],
                             outputs[2][-4][:, 0, ...]), -1)  # torch.Size([1, 3072])
        outputs = self.dropout(outputs)
        # Output 1
        outputs_classifier = nn.Sigmoid()(self.classifier(outputs))  # [%, %, %, %, %]
        # Output 2
        outputs_regressor = self.regressor(outputs).reshape(-1, 5, 4)  # 5 topic 4 aspect
        return outputs_classifier, outputs_regressor


def pred_to_label(outputs_classifier, outputs_regressor):
    result = np.zeros((outputs_classifier.shape[0], 5))  # [[0. 0. 0. 0. 0. ]]
    mask = (outputs_classifier >= 0.5)
    result[mask] = outputs_regressor[mask]
    return result


class ModelInference(nn.Module):
    def __init__(self, model_path, tokenizer, rdrsegmenter, checkpoint="vinai/phobert-base", device="cpu"):
        super(ModelInference, self).__init__()
        self.preprocess = Preprocess(tokenizer, rdrsegmenter)
        self.model = Analysis(checkpoint)
        self.device = device
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)
        self.model.to(device)

    def predict(self, sample):
        self.model.eval()
        with torch.no_grad():
            sample = self.preprocess.tokenize(sample)
            inputs = {"input_ids": sample["input_ids"].to(self.device),  # Clean input, segment, tokenize
                      "attention_mask": sample["attention_mask"].to(self.device)}
            outputs_classifier, outputs_regressor = self.model(**inputs)  # Predict
            # print(outputs_regressor)
            outputs_classifier = outputs_classifier.cpu().numpy()  # Convert to numpy array
            outputs_regressor = outputs_regressor.cpu().numpy()  # Convert to numpy array
            outputs_regressor = outputs_regressor.argmax(axis=-1) + 1  # Get argmax each topic is index (aspect [1,4])
            # print(outputs_regressor)
            outputs = pred_to_label(outputs_classifier, outputs_regressor)  # Convert output to label
        return outputs
