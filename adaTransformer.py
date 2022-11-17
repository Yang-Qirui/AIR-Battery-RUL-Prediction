import copy
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def cosine(source, target):
    source, target = source.mean(0), target.mean(0)
    cos = nn.CosineSimilarity(dim=0)
    loss = cos(source, target)
    return loss.mean()


class Autoencoder(nn.Module):
    def __init__(self, input_size=16, hidden_dim=8, noise_level=0.001):
        super(Autoencoder, self).__init__()
        self.input_size, self.hidden_dim, self.noise_level = input_size, hidden_dim, noise_level
        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_size)

    def encoder(self, x):
        x = self.fc1(x)
        h1 = F.relu(x)
        return h1

    def mask(self, x):
        corrupted_x = x + self.noise_level * torch.randn_like(x)
        return corrupted_x

    def decoder(self, x):
        h2 = self.fc2(x)
        return h2

    def forward(self, x):
        out = self.mask(x)
        encode = self.encoder(out)
        decode = self.decoder(encode)
        return encode, decode


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=16):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class Transformer(nn.Module):
    def __init__(self, output_size, feature_size=16, hidden_dim=32, num_layers=1, nhead=8, dropout=0.0, noise_level=0.01):
        super(Transformer, self).__init__()
        self.auto_hidden = int(feature_size/2)
        # # Auto-encoder
        input_size = self.auto_hidden
        self.autoencoder = Autoencoder(
            input_size=feature_size, hidden_dim=self.auto_hidden, noise_level=noise_level)
        # input_size = feature_size
        self.pos = PositionalEncoding(d_model=input_size)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.encoding = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers)
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=input_size, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.decoding = nn.TransformerDecoder(
            decoder_layers, num_layers=num_layers)
        self.linear1 = nn.Linear(input_size, output_size)
        # self.linear2 = nn.Linear(int(input_size/2), 1)

    def forward(self, x):
        batch_size, feature_num, feature_size = x.shape
        encode, decode = self.autoencoder(
            x.reshape(batch_size, -1))  # batch_size*seq_len
        out = encode.reshape(batch_size, -1, self.auto_hidden)
        # decode = x.reshape(batch_size, -1)
        # out = x
        out = self.pos(out)
        # shape (1, batch_size, feature_size)
        list_encode = out
        list_encoding = []
        for layer in self.encoding.layers:
            list_encode = layer(list_encode)
            list_encoding.append(list_encode)
        out = out.reshape(1, batch_size, -1)  # (1, batch_size, feature_size)
        out = self.encoding(out)
        out = out.reshape(batch_size, -1)    # (batch_size, hidden_dim)
        out = self.linear1(out)              # (batch_size, 1)

        return out, decode, list_encoding

    def adapt_encoding_weight(self, list_encoding, weight_mat=None):
        loss_all = torch.zeros(1).cuda()
        len_seq = list_encoding[0].shape[1]
        num_layers = len(list_encoding)
        if weight_mat is None:
            weight = (1.0 / len_seq *
                      torch.ones(num_layers, len_seq)).cuda()
        else:
            weight = weight_mat
        dist_mat = torch.zeros(num_layers, len_seq).cuda()
        for i in range(len(list_encoding)):
            data = list_encoding[i]
            # print("data_shape", data.shape)
            data_s = data[0:len(data)//2]
            data_t = data[len(data)//2:]
            # if train_type == 'last':
            #     loss_all = loss_all + 1 - cosine(data_s[:, -1, :],data_t[:, -1, :])
            # elif train_type == "all":
            for j in range(data_s.shape[1]):
                # print("data", data_s[:, j, :], data_t[:, j, :])
                loss_transfer = 1 - cosine(
                    data_s[:, j, :], data_t[:, j, :])
                # print("loss_transfer",loss_transfer)
                loss_all = loss_all + weight[i, j] * loss_transfer
                dist_mat[i, j] = loss_transfer
            # else:
            #     print("adapt loss error!")
        return loss_all, dist_mat, weight

    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-12
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * \
            (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1,
                                                                      len(weight_mat[0]))
        return weight_mat
