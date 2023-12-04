import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from utils import pred_to_label
from preprocessing.NewsPreprocessing import Preprocess


class CustomModelSoftmax(nn.Module):
    def __init__(self, checkpoint):
        """
        Custom PyTorch model for a multi-task learning scenario with softmax activation.

        Args:
            checkpoint (str): Pretrained model checkpoint name.
        """
        super(CustomModelSoftmax, self).__init__()
        # Load pretrained model with attention and hidden states output
        self.model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint,
                                                                      output_attentions=True,
                                                                      output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        # Classifier for classification task
        self.classifier = nn.Linear(768 * 4, 6)
        # Regressor for regression task
        self.regressor = nn.Linear(768 * 4, 24)

    def forward(self, input_ids=None, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Predicted outputs for classification and regression tasks.
        """
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        # Concatenate the last four hidden states
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


class ModelInference(nn.Module):
    def __init__(self, tokenizer, rdrsegmenter, model_path, checkpoint="vinai/phobert-base", device="cpu"):
        """
        Inference module for the model.

        Args:
            tokenizer: Tokenizer for processing input.
            rdrsegmenter: Segmenter for input.
            model_path (str): Path to the trained model.
            checkpoint (str): Pretrained model checkpoint name.
            device (str): Device on which the model will run.
        """
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
        """
        Make predictions with the trained model.

        Args:
            sample (str): Input text sample.

        Returns:
            dict: Predicted labels for classification and regression tasks.
        """
        self.model.eval()
        with torch.no_grad():
            # Clean input, segment, and tokenize
            sample = self.preprocess.tokenize(sample)
            inputs = {"input_ids": sample["input_ids"].to(self.device),
                      "attention_mask": sample["attention_mask"].to(self.device)}

            # Predict
            outputs_classifier, outputs_regressor = self.model(**inputs)

            # Convert to numpy array
            outputs_classifier = outputs_classifier.cpu().numpy()
            outputs_regressor = outputs_regressor.cpu().numpy()

            # Get argmax for each aspect in the regression task
            outputs_regressor = outputs_regressor.argmax(axis=-1) + 1

            # Convert output to label
            outputs = pred_to_label(outputs_classifier, outputs_regressor)
        return outputs
