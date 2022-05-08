import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from utils.process import normalize_adj


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1,
                                                                                                   2 * self.out_features)
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' ( ' + str(self.in_features) + '->' + str(self.out_features) + ' ) '


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers=2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nheads = nheads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                for j in range(self.nheads):
                    self.add_module('attention_{}_{}'.format(i + 1, j),
                                    GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True))
        self.out_attn = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        input = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                temp = []
                x = F.dropout(x, self.dropout, training=self.training)
                cur_input = x
                for j in range(self.nheads):
                    temp.append(self.__getattr__('attenion_{}_{}'.format(i + 1, j))(x, adj))
                x = torch.cat(temp, dim=2) + cur_input
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_attn(x, adj))
        return x + input


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.__args = args
        self.__encoder = LSTMEncoder(
            self.__args.word_embedding_dim,
            self.__args.encoder_hidden_dim,
            self.__args.dropout_rate
        )
        self.__attention = SelfAttention(
            self.__args.word_embedding_dim,
            self.__args.attention_hidden_dim,
            self.__args.attention_output_dim,
            self.__args.dropout_rate
        )

    def forward(self, word_tensor, seq_lens):
        lstm_hidden = self.__encoder(word_tensor, seq_lens)
        attention_hidden = self.__attention(word_tensor, seq_lens)
        hidden = torch.cat([attention_hidden, lstm_hidden], dim=2)
        return hidden


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        self.__embedding_dim = embedding_dim
        if bidirectional:
            self.__hidden_dim = hidden_dim // 2
        else:
            self.__hidden_dim = hidden_dim
        self.__dropout_rate = dropout_rate
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        dropout_text = self.__dropout_layer(embedded_text)
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hidden, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hidden, _ = pad_packed_sequence(lstm_hidden, batch_first=True)
        return padded_hidden


class QKVAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ), dim=-1) / math.sqrt(self.__hidden_dim)
        force_tensor = torch.matmul(score_tensor, linear_value)
        force_tensor = self.__dropout_layer(force_tensor)
        return force_tensor


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(dropout_x, dropout_x, dropout_x)
        return attention_x


class LSTMDecoder(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, dropout_rate, embedding_dim=None, extra_dim=None):
        super(LSTMDecoder, self).__init__()
        self.__args = args
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(
                torch.randn(1, self.__embedding_dim),
                requires_grad=True
            )

        self.__dropout_layer = nn.Dropout(self.__args.dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__args.encoder_hidden_dim + self.__args.attention_output_dim + self.__args.word_embedding_dim + self.__args.word_embedding_dim,
            hidden_size=self.__args.encoder_hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1
        )
        self.__global_graph = GAT(
            self.__hidden_dim,
            self.__args.decoder_gat_hidden_dim,
            self.__hidden_dim,
            self.__args.gat_dropout_rate,
            self.__args.alpha,
            self.__args.n_heads,
            self.__args.n_layers_decoder_global
        )
        self.__linear_layer = nn.Sequential(
            nn.Linear(self.__hidden_dim, self.__hidden_dim),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__hidden_dim, self.__output_dim)
        )
        self.__linear = nn.Linear(self.__args.encoder_hidden_dim, self.__args.word_embedding_dim)

    def forward(self, encoded_hidden, seq_lens, intent_lstm_out, intent_h_last, intent_slot_adj, forced_input=None):
        input_tensor = encoded_hidden
        output_tensor_list, sent_start_pos = [], 0
        intent_lstm_out = self.__linear(intent_lstm_out)
        if forced_input is not None:
            forced_tensor = self.__embedding_layer(forced_input)[:, :-1]
            prev_tensor = torch.cat((self.__init_tensor.unsqueeze(0).repeat(len(forced_tensor), 1, 1), forced_tensor),
                                    dim=1)
            combined_input = torch.cat([input_tensor, intent_lstm_out, prev_tensor], dim=2)
            dropout_input = self.__dropout_layer(combined_input)
            packed_text = pack_padded_sequence(dropout_input, seq_lens, batch_first=True)
            slot_lstm_out, (h_last, c_last) = self.__lstm_layer(packed_text)
            padded_hidden, _ = pad_packed_sequence(slot_lstm_out, batch_first=True)
            for sent_i in range(0, len(seq_lens)):
                if intent_slot_adj is not None:
                    lstm_out_i = torch.cat((padded_hidden[sent_i][:seq_lens[sent_i]].unsqueeze(1),
                                            intent_h_last.squeeze(0)[sent_i].repeat(seq_lens[sent_i], 1, 1)), dim=1)
                    lstm_out_i = self.__global_graph(lstm_out_i,
                                                     intent_slot_adj[sent_i].unsqueeze(0).repeat(seq_lens[sent_i], 1,
                                                                                                 1))[:, 0]
                else:
                    lstm_out_i = padded_hidden[sent_i][:seq_lens[sent_i]]
                linear_out = self.__linear_layer(lstm_out_i)
                output_tensor_list.append(linear_out)
        else:
            prev_tensor = self.__init_tensor.unsqueeze(0).repeat(len(seq_lens), 1, 1)
            last_h, last_c = None, None
            for word_i in range(seq_lens[0]):
                combined_input = torch.cat(
                    (input_tensor[:, word_i].unsqueeze(1), intent_lstm_out[:, word_i].unsqueeze(1), prev_tensor), dim=2)
                dropout_input = self.__dropout_layer(combined_input)
                if last_h is None and last_c is None:
                    lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                else:
                    lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input, (last_h, last_c))
                if intent_slot_adj is not None:
                    lstm_out = torch.cat((lstm_out,
                                          intent_h_last.squeeze(0)[0].repeat(len(lstm_out), 1, 1)), dim=1)
                    lstm_out = self.__global_graph(lstm_out, intent_slot_adj)[:, 0]

                lstm_out = self.__linear_layer(lstm_out.squeeze(1))
                output_tensor_list.append(lstm_out)

                _, index = lstm_out.topk(1, dim=1)
                prev_tensor = self.__embedding_layer(index.squeeze(1)).unsqueeze(1)
            output_tensor = torch.stack(output_tensor_list)
            output_tensor_list = [output_tensor[:length, i] for i, length in enumerate(seq_lens)]

        return torch.cat(output_tensor_list, dim=0)


class ModelManager(nn.Module):
    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()
        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        self.__word_embedding = nn.Embedding(self.__num_word, self.__args.word_embedding_dim)
        self.G_encoder = Encoder(args)
        self.__intent_decoder = IntentDecoder(args, num_intent)
        self.__intent_lstm = LSTMEncoder(
            self.__args.encoder_hidden_dim,
            self.__args.encoder_hidden_dim,
            self.__args.dropout_rate
        )
        self.__slot_decoder = LSTMDecoder(
            args,
            self.__args.encoder_hidden_dim,
            self.__args.encoder_hidden_dim,
            self.__num_slot,
            self.__args.dropout_rate,
            embedding_dim=self.__args.slot_embedding_dim
        )

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))
        print('\tword embedding dimension:				    {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				    {};'.format(self.__args.encoder_hidden_dim))
        print('\tdimension of intent embedding:		    	{};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot decoder hidden:  	    {};'.format(self.__args.slot_decoder_hidden_dim))
        print('\thidden dimension of self-attention:        {};'.format(self.__args.attention_hidden_dim))
        print('\toutput dimension of self-attention:        {};'.format(self.__args.attention_output_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')


    def generate_intent_slot_adj_gat(self, batch):
        adj = torch.cat([torch.eye(1 + 1).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in range(2):
                adj[i, j, 0] = 1
        if self.__args.row_normalized:
            adj = normalize_adj(adj)
        if self.__args.gpu:
            adj = adj.cuda()
        return adj

    def forward(self, text, seq_lens, n_predicts=None, forced_slot=None):
        word_tensor = self.__word_embedding(text)
        g_hidden = self.G_encoder(word_tensor, seq_lens)
        pred_intent, intent_lstm_out, intent_h_last = self.__intent_decoder(g_hidden, seq_lens)
        intent_index = (torch.sigmoid(pred_intent) > self.__args.threshold).nonzero()
        intent_slot_adj = self.generate_intent_slot_adj_gat(len(pred_intent))
        pred_slot = self.__slot_decoder(
            g_hidden, seq_lens, intent_lstm_out,  intent_h_last,
            forced_input=forced_slot,
            intent_slot_adj=intent_slot_adj
        )
        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), pred_intent
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)

            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()


class IntentDecoder(nn.Module):
    def __init__(self, args, num_intent):
        super(IntentDecoder, self).__init__()
        self.__args = args
        self.__num_intent = num_intent
        self.__intent_decoder = nn.Sequential(
            nn.Linear(self.__args.encoder_hidden_dim, self.__args.encoder_hidden_dim),
            nn.LeakyReLU(self.__args.alpha),
            nn.Linear(self.__args.encoder_hidden_dim, self.__num_intent)
        )
        self.__dropout_layer = nn.Dropout(self.__args.dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            hidden_size=self.__args.encoder_hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__args.dropout_rate,
            num_layers=1
        )

    def forward(self, encoder_hidden, seq_lens):
        dropout_text = self.__dropout_layer(encoder_hidden)
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hidden, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hidden, _ = pad_packed_sequence(lstm_hidden, batch_first=True)
        logits_intent = self.__intent_decoder(h_last.squeeze(0))
        return logits_intent, padded_hidden, h_last





















