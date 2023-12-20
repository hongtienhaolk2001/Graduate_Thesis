import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from ..utils import pred_to_label
from ..preprocessing.NewsPreprocessing import Preprocess


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


class ModelInference(nn.Module):
    def __init__(self, tokenizer, rdrsegmenter, model_path, checkpoint="vinai/phobert-base", device="cpu"):
        super(ModelInference, self).__init__()
        # Preprocessing module
        self.preprocess = Preprocess(tokenizer, rdrsegmenter)
        # Load the custom model
        self.model = CustomModelSoftmax(checkpoint)
        self.device = device
        # Load the trained model's weights
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.model.to(device)

    def predict(self, sample):
        self.model.eval()
        with torch.no_grad():
            # Clean input, segment, and tokenize
            sample = self.preprocess.tokenize(sample)
            inputs = {"input_ids": sample["input_ids"].to(self.device),
                      "attention_mask": sample["attention_mask"].to(self.device)}

            # Predict
            outputs_classifier = self.model(**inputs)

            # Convert to numpy array
            outputs_classifier = outputs_classifier.cpu().numpy()

            # Convert output to label
            outputs = pred_to_label(outputs_classifier)
        return outputs
