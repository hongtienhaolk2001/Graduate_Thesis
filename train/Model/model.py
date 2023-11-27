# import numpy as np
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoConfig
#
# import CustomModel
# import CustomModelMultiHead
# import CustomModelMultiHeadRegressor
# import CustomModelSoftmax
# from utils import pred_to_label
# from ...preprocessing.NewsPreprocessing import Preprocess
#
#
# # class ModelInference(nn.Module):
# #     def __init__(self, tokenizer, rdrsegmenter, model_path, checkpoint="vinai/phobert-base", device="cpu"):
# #         super(ModelInference, self).__init__()
# #         self.preprocess = Preprocess(tokenizer, rdrsegmenter)
# #         self.model = CustomModelSoftmax(checkpoint)
# #         self.device = device
# #         self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
# #         self.model.to(device)
# #
# #     def predict(self, sample):
# #         self.model.eval()
# #         with torch.no_grad():
# #             # Clean input, segment and tokenize
# #             sample = self.preprocess.tokenize(sample)
# #             inputs = {"input_ids": sample["input_ids"].to(self.device),
# #                       "attention_mask": sample["attention_mask"].to(self.device)}
# #             # Predict
# #             outputs_classifier, outputs_regressor = self.model(**inputs)
# #             # Convert to numpy array
# #             outputs_classifier = outputs_classifier.cpu().numpy()
# #             outputs_regressor = outputs_regressor.cpu().numpy()
# #             # Get argmax each aspect
# #             outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
# #             # Convert output to label
# #             outputs = pred_to_label(outputs_classifier, outputs_regressor)
# #         return outputs
#
#
# def score_to_tensor(score):
#     tensor = np.zeros((score.shape[0], 6, 6))
#     mask_up = np.ceil(score).reshape(-1).astype(np.int16)
#     mask_down = np.floor(score).reshape(-1).astype(np.int16)
#     xv, yv = np.meshgrid(np.arange(score.shape[0]), np.arange(6))
#     y = yv.T.reshape(-1).astype(np.int16)
#     x = xv.T.reshape(-1).astype(np.int16)
#     score_up = (score - np.floor(score)).reshape(-1)
#     score_down = (1 - score_up).reshape(-1)
#     tensor[x, y, mask_up] = score_up
#     tensor[x, y, mask_down] = score_down
#     tensor[:, :, 1] = tensor[:, :, 0] + tensor[:, :, 1]
#     return tensor[:, :, 1:]
#
#
# class CustomModelRegressor(nn.Module):
#     def __init__(self, checkpoint, num_outputs):
#         super(CustomModelRegressor, self).__init__()
#         self.num_outputs = num_outputs
#         self.model = AutoModel.from_pretrained(checkpoint,
#                                                config=AutoConfig.from_pretrained(checkpoint,
#                                                                                  output_attentions=True,
#                                                                                  output_hidden_states=True))
#         for parameter in self.model.parameters():
#             parameter.require_grad = False
#         self.dropout = nn.Dropout(0.1)
#         self.output1 = nn.Linear(768 * 4, 6)
#
#     def forward(self, input_ids=None, attention_mask=None):
#         outputs = self.model(input_ids=input_ids,
#                              attention_mask=attention_mask)
#         outputs = torch.cat((outputs[2][-1][:, 0, ...],
#                              outputs[2][-2][:, 0, ...],
#                              outputs[2][-3][:, 0, ...],
#                              outputs[2][-4][:, 0, ...]), -1)
#         outputs = self.dropout(outputs)
#         outputs = self.output1(outputs)
#         outputs = nn.Sigmoid()(outputs) * 5
#         return outputs
#
#
# class CustomModelClassifier(nn.Module):
#     def __init__(self, checkpoint, num_outputs):
#         super(CustomModelClassifier, self).__init__()
#         self.num_outputs = num_outputs
#         self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint,
#                                                                                              output_attentions=True,
#                                                                                              output_hidden_states=True))
#         for parameter in self.model.parameters():
#             parameter.require_grad = False
#         self.dropout = nn.Dropout(0.1)
#         self.output1 = nn.Linear(768 * 4, 30)
#
#     def forward(self, input_ids=None, attention_mask=None):
#         outputs = self.model(input_ids=input_ids,
#                              attention_mask=attention_mask)
#         outputs = torch.cat((outputs[2][-1][:, 0, ...], outputs[2][-2][:, 0, ...], outputs[2][-3][:, 0, ...],
#                              outputs[2][-4][:, 0, ...]), -1)
#         outputs = self.dropout(outputs)
#         outputs = self.output1(outputs)
#         return outputs
#
#
# class ModelEnsemble(nn.Module):
#     def __init__(self, tokenizer, rdrsegmenter, model_path1, model_path2, model_path3, model_path4,
#                  checkpoint="vinai/phobert-base", device="cpu"):
#         super(ModelEnsemble, self).__init__()
#         self.preprocess = Preprocess(tokenizer, rdrsegmenter)
#         self.model1 = CustomModelSoftmax(checkpoint)
#         self.model2 = CustomModel(checkpoint)
#         self.model3 = CustomModelMultiHead(checkpoint)
#         self.model4 = CustomModelMultiHeadRegressor(checkpoint)
#         self.device = device
#         self.model1.load_state_dict(torch.load(model_path1, map_location=torch.device(device)))
#         self.model1.to(device)
#         self.model2.load_state_dict(torch.load(model_path2, map_location=torch.device(device)))
#         self.model2.to(device)
#         self.model3.load_state_dict(torch.load(model_path3, map_location=torch.device(device)))
#         self.model3.to(device)
#         self.model4.load_state_dict(torch.load(model_path4, map_location=torch.device(device)))
#         self.model4.to(device)
#
#     def predict(self, sample):
#         self.model1.eval()
#         self.model2.eval()
#         self.model3.eval()
#         with torch.no_grad():
#             sample = self.preprocess.tokenize(sample)
#             inputs = {"input_ids": sample["input_ids"].to(self.device),
#                       "attention_mask": sample["attention_mask"].to(self.device)}
#             outputs_classifier1, outputs_regressor1 = self.model1(**inputs)
#             outputs_classifier2, outputs_regressor2 = self.model2(**inputs)
#             outputs_classifier3, outputs_regressor3 = self.model3(**inputs)
#             outputs_classifier4, outputs_regressor4 = self.model4(**inputs)
#
#             outputs_classifier1 = outputs_classifier1.cpu().numpy()
#             outputs_regressor1 = outputs_regressor1.cpu().numpy()
#             outputs_classifier2 = outputs_classifier2.cpu().numpy()
#             outputs_regressor2 = outputs_regressor2.cpu().numpy()
#             outputs_classifier3 = outputs_classifier3.cpu().numpy()
#             outputs_regressor3 = outputs_regressor3.cpu().numpy()
#             outputs_classifier4 = outputs_classifier4.cpu().numpy()
#             outputs_regressor4 = outputs_regressor4.cpu().numpy()
#
#             outputs_regressor2 = score_to_tensor(outputs_regressor2)
#             outputs_regressor4 = score_to_tensor(outputs_regressor4)
#
#             outputs_regressor = (nn.Softmax(dim=-1)(
#                 torch.tensor(outputs_regressor1)).numpy() + outputs_regressor2 + nn.Softmax(dim=-1)(
#                 torch.tensor(outputs_regressor3)).numpy() + outputs_regressor4) / 4
#             outputs_classifier = (outputs_classifier1 + outputs_classifier2 + outputs_classifier3 + outputs_classifier4) / 4
#             outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
#             outputs = pred_to_label(outputs_classifier, outputs_regressor)[0].astype(np.int32)
#         return outputs.tolist()
#
#
# # if __name__ == "__main__":
# #     rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
# #     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
# #     model = ModelInference(tokenizer, rdrsegmenter, 'weights/model_softmax_v4.pt')
# #     # model = ModelEnsemble(tokenizer, rdrsegmenter, 'weights/model_softmax_v2_submit.pt', 'weights/model_regress_v2_submit.pt')
# #     print(model.predict(""))
