from transformers import AutoModel, AutoConfig

from CustomTransformer import *


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs


class CustomModelMultiHeadRegressor(nn.Module):
    def __init__(self, checkpoint):
        super(CustomModelMultiHeadRegressor, self).__init__()
        self.model = AutoModel.from_config(
            AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 6)
        self.regressor = nn.Linear(768, 6)
        self.encoderlayer = EncoderLayer(768, 64, 64, 12, 3096)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.dropout(outputs[0])
        outputs = self.encoderlayer(outputs, attention_mask.type(torch.bool))  # N, S, D
        outputs = outputs[:, 0, :]
        outputs_classifier = self.classifier(outputs)
        outputs_regressor = self.regressor(outputs)
        outputs_classifier = nn.Sigmoid()(outputs_classifier)
        outputs_regressor = nn.Sigmoid()(outputs_regressor) * 5
        return outputs_classifier, outputs_regressor
