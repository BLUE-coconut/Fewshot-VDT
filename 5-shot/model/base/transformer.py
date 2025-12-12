# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer utils."""

import math
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
import mindspore.numpy as mnp

def get_fixed_sin_cos_encodings(d_model, max_len):

    assert d_model % 2 == 0
    position = ops.expand_dims(ops.arange(max_len, dtype=mindspore.float32), 1)

    div_term = ops.exp(
        ops.arange(0, d_model, 2, dtype=mindspore.float32) * (-math.log(10000.0) / d_model)
    )

    pe = ops.zeros((max_len, d_model), mindspore.float32)

    pe[:, 0:d_model:2] = ops.sin(position * div_term)
    pe[:, 1:d_model:2] = ops.cos(position * div_term)
    
    return pe


class AbsolutePositionalEncoding(nn.Cell):

    def __init__(self, max_len, d_model, trainable=False):
        super(AbsolutePositionalEncoding, self).__init__()
        self.max_len = max_len
        self.trainable = trainable
        if trainable:
            self.pe = nn.Embedding(max_len, d_model)
        else:
            fixed_pe = get_fixed_sin_cos_encodings(d_model, max_len)
            self.pe = fixed_pe

    def construct(self, x):

        batch_size = x.shape[0]
        actual_len = x.shape[1]
        
        if self.trainable:
            pe = self.pe.embedding_table
        else:
            pe = self.pe

        pe_tiled = ops.tile(ops.expand_dims(pe, 0), (batch_size, 1, 1))

        return ops.slice(pe_tiled, (0, 0, 0), (batch_size, actual_len, d_model))

    def get_pe(self, position):
        if self.trainable:
            pe = self.pe.embedding_table
        else:
            pe = self.pe

        return pe[position]


class RelativePositionalEncoding(nn.Cell):

    def __init__(
        self,
        max_relative_position,
        d_model,
        trainable=False,
        cross_attn=False,
    ):
        super(RelativePositionalEncoding, self).__init__()
        self.max_relative_position = max_relative_position
        self.trainable = trainable
        self.cross_attn = cross_attn
        self.num_embeddings = (
            (max_relative_position * 2 + 1)
            if not cross_attn
            else (max_relative_position + 1)
        )
        if trainable:
            self.embeddings_table = nn.Embedding(self.num_embeddings, d_model)
        else:
            fixed_encodings = get_fixed_sin_cos_encodings(d_model, max_relative_position * 2 + 1)
            self.embeddings_table = fixed_encodings

    def construct(self, length_q, length_k):
        if self.trainable:
            embeddings_table = self.embeddings_table.embedding_table
        else:
            embeddings_table = self.embeddings_table

        if self.cross_attn:
            k_arange = ops.expand_dims(ops.arange(length_k - 1, -1, -1, dtype=mindspore.int64), 0)
            q_arange = ops.expand_dims(ops.arange(length_q, dtype=mindspore.int64), 1)
            distance_mat = k_arange + q_arange
        else:
            k_arange = ops.expand_dims(ops.arange(length_k, dtype=mindspore.int64), 0)
            q_arange = ops.expand_dims(ops.arange(length_q, dtype=mindspore.int64), 1)
            distance_mat = k_arange - q_arange

        distance_mat_clipped = ops.clip_by_value(
            distance_mat, 
            Tensor([-self.max_relative_position], mindspore.int64), 
            Tensor([self.max_relative_position], mindspore.int64)
        )
        
        if not self.cross_attn:
            distance_mat_clipped = distance_mat_clipped + self.max_relative_position

        final_mat = distance_mat_clipped

        embeddings = embeddings_table[final_mat]

        return embeddings


class UnalignedRelativePositionalEncoding(RelativePositionalEncoding):
    """Unaligned relative positional encoding (MindSpore)."""

    def __init__(self, *args, **kwargs):
        super(UnalignedRelativePositionalEncoding, self).__init__(*args, **kwargs)

    def construct(self, length_q, length_k):
        if self.trainable:
            embeddings_table = self.embeddings_table.embedding_table
        else:
            embeddings_table = self.embeddings_table

        if self.cross_attn:
            assert length_q == length_k
            k_arange = ops.expand_dims(ops.arange(length_k - 1, -1, -1, dtype=mindspore.int64), 0)
            q_arange = ops.expand_dims(ops.arange(length_q, dtype=mindspore.int64), 1)
            distance_mat = k_arange + q_arange - (length_q - 1)
            distance_mat = ops.clip_by_value(
                distance_mat, 
                Tensor([0], mindspore.int64), 
                Tensor([self.max_relative_position], mindspore.int64)
            )
        else:
            k_arange = ops.expand_dims(ops.arange(length_k, dtype=mindspore.int64), 0)
            q_arange = ops.expand_dims(ops.arange(length_q, dtype=mindspore.int64), 1)
            distance_mat = k_arange - q_arange
            
        distance_mat_clipped = ops.clip_by_value(
            distance_mat, 
            Tensor([-self.max_relative_position], mindspore.int64), 
            Tensor([self.max_relative_position], mindspore.int64)
        )
        
        if not self.cross_attn:
            distance_mat_clipped = distance_mat_clipped + self.max_relative_position
            
        final_mat = distance_mat_clipped
        embeddings = embeddings_table[final_mat]

        return embeddings


class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.layer = nn.LayerNorm((normalized_shape,), epsilon=eps)

    def construct(self, x):
        return self.layer(x)


class Attention(nn.Cell):
    """Multi head attention in MindSpore, without ops.Tril."""

    def __init__(self, positional_encoding_k=None, positional_encoding_v=None):
        super(Attention, self).__init__()
        self.positional_encoding_k = positional_encoding_k
        self.positional_encoding_v = positional_encoding_v
        self.matmul = ops.BatchMatMul()
        self.einsum1 = ops.Einsum('b h q d, q k d -> b h q k')
        self.einsum2 = ops.Einsum('b h q v, q v d -> b h q d')
        self.softmax = ops.Softmax(axis=-1)
        self.select = ops.Select()
        self.ones_like = ops.OnesLike()
        self.sqrt = ops.Sqrt()

    def construct(self, query, key, value, mask=None, dropout=None, one_direction=False):
        scores = self.matmul(query, key.transpose(0, 1, 3, 2))

        if self.positional_encoding_k is not None:
            bigr_k = self.positional_encoding_k(query.shape[2], key.shape[2])
            scores = scores + self.einsum1(query, bigr_k)
        scores = scores / self.sqrt(Tensor(query.shape[-1], dtype=mstype.float32))

        if mask is not None:
            neg_inf = Tensor(-1e9, dtype=scores.dtype)
            scores = self.select(mask, scores, neg_inf)

        if one_direction:
            seq_len = query.shape[2]
            tril_mask = mnp.tril(mnp.ones((seq_len, seq_len), dtype=mstype.float32))
            tril_mask = tril_mask.expand_as(scores)
            tril_mask = tril_mask.astype(mstype.bool_)
            neg_inf = Tensor(-1e9, dtype=scores.dtype)
            neg_inf = mnp.full(scores.shape, -1e9, dtype=mstype.float32)
            scores = self.select(tril_mask, scores, neg_inf)

        p_attn = self.softmax(scores)

        if dropout is not None:
            p_attn = dropout(p_attn)

        output = self.matmul(p_attn, value)

        if self.positional_encoding_v is not None:
            bigr_v = self.positional_encoding_v(query.shape[2], value.shape[2])
            output = output + self.einsum2(p_attn, bigr_v)

        return output, p_attn



class MultiHeadedAttention(nn.Cell):
    """Multihead attention block (MindSpore)."""

    def __init__(
        self,
        num_heads,
        d_model,
        head_size=None,
        dropout=0.0,
        positional_encoding_k=None,
        positional_encoding_v=None,
        final_layer=False,
    ):
        super(MultiHeadedAttention, self).__init__()

        if d_model % num_heads != 0:
            raise AssertionError(f'd_model: {d_model} and num_heads: {num_heads}')

        self.num_heads = num_heads
        if head_size is not None:
            self.head_size = head_size
        else:
            self.head_size = d_model // num_heads
        
        total_dim = self.num_heads * self.head_size

        self.linear_layers = nn.CellList(
            [nn.Dense(d_model, total_dim) for _ in range(3)]
        )
        self.attention = Attention(positional_encoding_k, positional_encoding_v)
        self.dropout = nn.Dropout(keep_prob=1-dropout)
        
        if final_layer:
            self.final_layer = nn.Dense(total_dim, d_model)
        
        self.layer_norm = LayerNorm(d_model)

    def construct(self, query, key, value, mask=None, one_direction=True, prefix=None):
        bsz, ch, h, w = query.shape
        seq_len = h * w

        query_flat = ops.reshape(ops.transpose(query, (0, 2, 3, 1)), (bsz, -1, ch))
        key_flat = ops.reshape(ops.transpose(key, (0, 2, 3, 1)), (bsz, -1, ch))
        value_flat = ops.reshape(ops.transpose(value, (0, 2, 3, 1)), (bsz, -1, ch))

        query_, key_, value_ = [], [], []
        
        for layer, x in zip(self.linear_layers, (query_flat, key_flat, value_flat)):
            projected = layer(x)
            
            reshaped = ops.reshape(
                projected, (bsz, seq_len, self.num_heads, self.head_size)
            )
            transposed = ops.transpose(reshaped, (0, 2, 1, 3))
            
            if layer == self.linear_layers[0]:
                query_ = transposed
            elif layer == self.linear_layers[1]:
                key_ = transposed
            else:
                value_ = transposed

        if prefix is not None:
            prefix_key_ = ops.expand_dims(prefix[0], 0)
            prefix_value_ = ops.expand_dims(prefix[1], 0)

            prefix_key_ = ops.transpose(ops.tile(prefix_key_, (bsz, 1, 1, 1)), (0, 2, 1, 3))
            prefix_value_ = ops.transpose(ops.tile(prefix_value_, (bsz, 1, 1, 1)), (0, 2, 1, 3))

            key_ = ops.cat((prefix_key_, key_), axis=2)
            value_ = ops.cat((prefix_value_, value_), axis=2)

            prefix_mask = ops.ones((1, 1, 1, prefix_key_.shape[2]), dtype=mask.dtype)
            prefix_mask = ops.tile(
                prefix_mask, (mask.shape[0], mask.shape[1], mask.shape[2], 1)
            )
            mask = ops.cat((prefix_mask, mask), axis=3)

        x, _ = self.attention(
            query_,
            key_,
            value_,
            mask=mask,
            dropout=self.dropout,
            one_direction=one_direction,
        )

        x = ops.transpose(x, (0, 2, 1, 3))
        total_dim = self.num_heads*self.head_size
        x = ops.reshape(x, (bsz, seq_len, total_dim))

        if hasattr(self, 'final_layer'):
            x = self.final_layer(x)

        residual_x = x + query_flat
        normed_x = self.layer_norm(residual_x)

        final_output = ops.reshape(ops.transpose(normed_x, (0, 2, 1)), (bsz, ch, h, w))
        
        return final_output


class PositionwiseFeedForward(nn.Cell):
    """Position-wise feed forward module (MindSpore)."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = LayerNorm(d_model)

        self.transpose = ops.Transpose()
        self.add = ops.Add()

    def construct(self, x):
        residual = x

        x_transposed = self.transpose(x, (0, 2, 1))
        x_ = self.dropout(self.activation(self.conv1(x_transposed)))

        output = self.transpose(self.dropout(self.conv2(x_)), (0, 2, 1))
        return self.layer_norm(self.add(output, residual))


class TransformerEncoderBlock(nn.Cell):
    """Transformer encoder block (MindSpore)."""

    def __init__(
        self,
        hidden,
        attn_heads,
        head_size,
        feed_forward_hidden,
        dropout,
        attn_dropout=0.1,
        self_positional_encoding_k=None,
        self_positional_encoding_v=None,
        final_layer=True,
        **kwargs,
    ):
        super(TransformerEncoderBlock, self).__init__()
        
        self.self_attention = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k,
            positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze(axis=-1)
        self.mul = ops.Mul()

    def construct(self, x, active_entries):

        active_entries_sq1 = self.expand_dims(active_entries, 1)
        active_entries_sq2 = self.expand_dims(active_entries, 2)

        active_entries_mask = active_entries

        mask_mat_unsq = self.mul(self.expand_dims(active_entries_mask, 1), self.expand_dims(active_entries_mask, 2))

        self_att_mask = self.expand_dims(self.squeeze(mask_mat_unsq), 1) 

        x = self.self_attention(x, x, x, self_att_mask, True)

        ffn_input = ops.reshape(ops.transpose(x, (0, 2, 3, 1)), (x.shape[0], x.shape[2]*x.shape[3], x.shape[1])) # (B, L, d_model)
        
        ffn_output = self.feed_forward(ffn_input)

        return ffn_output


class TransformerDecoderBlock(nn.Cell):
    """Transformer decoder block (MindSpore)."""

    def __init__(
        self,
        hidden,
        attn_heads,
        head_size,
        feed_forward_hidden,
        dropout,
        attn_dropout,
        self_positional_encoding_k=None,
        self_positional_encoding_v=None,
        cross_positional_encoding_k=None,
        cross_positional_encoding_v=None,
        final_layer=False,
        **kwargs,
    ):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attention = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k,
            positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )
        self.cross_attention = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=cross_positional_encoding_k,
            positional_encoding_v=cross_positional_encoding_v,
            final_layer=final_layer,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.expand_dims = ops.ExpandDims()
        self.mul = ops.Mul()

    def construct(self, x, encoder_x, active_entries, active_encoder_br):

        active_entries_mask = active_entries
        mask_mat_unsq = self.mul(self.expand_dims(active_entries_mask, 1), self.expand_dims(active_entries_mask, 2)) # (B, L_q, L_q, 1)
        self_att_mask = self.expand_dims(ops.squeeze(mask_mat_unsq, -1), 1)

        active_enc_sq1 = self.expand_dims(active_encoder_br, 1)
        active_dec_sq1 = self.expand_dims(active_entries_mask, 2)
        cross_mask_mat_unsq = self.mul(active_dec_sq1, active_enc_sq1)
        cross_att_mask = self.expand_dims(ops.squeeze(cross_mask_mat_unsq, -1), 1)

        x_mha = ops.transpose(self.expand_dims(x, 2), (0, 3, 2, 1))
        encoder_x_mha = ops.transpose(self.expand_dims(encoder_x, 2), (0, 3, 2, 1))

        x_att = self.self_attention(x_mha, x_mha, x_mha, self_att_mask, True)
        x_att = ops.squeeze(ops.transpose(x_att, (0, 3, 2, 1)), 2)

        x_cross = self.cross_attention(
            ops.transpose(self.expand_dims(x_att, 2), (0, 3, 2, 1)), 
            encoder_x_mha, encoder_x_mha, 
            cross_att_mask, False
        )
        x_cross = ops.squeeze(ops.transpose(x_cross, (0, 3, 2, 1)), 2)

        x = self.feed_forward(x_cross)
        
        return x


class TransformerMultiInputBlock(nn.Cell):
    """Transformer multiple input block (MindSpore)."""

    def __init__(
        self,
        hidden,
        attn_heads,
        head_size,
        feed_forward_hidden,
        dropout,
        attn_dropout,
        self_positional_encoding_k=None,
        self_positional_encoding_v=None,
        n_inputs=2,
        final_layer=False,
        disable_cross_attention=False,
        isolate_subnetwork='',
        **kwargs,
    ):
        super(TransformerMultiInputBlock, self).__init__()
        self.n_inputs = n_inputs
        self.disable_cross_attention = disable_cross_attention
        self.isolate_subnetwork = isolate_subnetwork

        attention_block_names = []

        self.self_attention_o = MultiHeadedAttention(
            num_heads=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k, positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )
        attention_block_names.append('self_attention_o')
        
        self.self_attention_t = MultiHeadedAttention(
            num_heads=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k, positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )
        attention_block_names.append('self_attention_t')

        if not disable_cross_attention:
            self.cross_attention_ot = MultiHeadedAttention(
                num_heads=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout,
                positional_encoding_k=self_positional_encoding_k, positional_encoding_v=self_positional_encoding_v,
                final_layer=final_layer,
            )
            attention_block_names.append('cross_attention_ot')
            self.cross_attention_to = MultiHeadedAttention(
                num_heads=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout,
                positional_encoding_k=self_positional_encoding_k, positional_encoding_v=self_positional_encoding_v,
                final_layer=final_layer,
            )
            attention_block_names.append('cross_attention_to')

        if n_inputs == 3:
            self.self_attention_v = MultiHeadedAttention(
                num_heads=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout,
                positional_encoding_k=self_positional_encoding_k, positional_encoding_v=self_positional_encoding_v,
                final_layer=final_layer,
            )
            attention_block_names.append('self_attention_v')
            
            if not disable_cross_attention:
                self.cross_attention_tv = MultiHeadedAttention(
                    num_heads=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout,
                    positional_encoding_k=self_positional_encoding_k, positional_encoding_v=self_positional_encoding_v,
                    final_layer=final_layer,
                )
                attention_block_names.append('cross_attention_tv')
                self.cross_attention_vt = MultiHeadedAttention(
                    num_heads=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout,
                    positional_encoding_k=self_positional_encoding_k, positional_encoding_v=self_positional_encoding_v,
                    final_layer=final_layer,
                )
                attention_block_names.append('cross_attention_vt')
                self.cross_attention_ov = MultiHeadedAttention(
                    num_heads=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout,
                    positional_encoding_k=self_positional_encoding_k, positional_encoding_v=self_positional_encoding_v,
                    final_layer=final_layer,
                )
                attention_block_names.append('cross_attention_ov')
                self.cross_attention_vo = MultiHeadedAttention(
                    num_heads=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout,
                    positional_encoding_k=self_positional_encoding_k, positional_encoding_v=self_positional_encoding_v,
                    final_layer=final_layer,
                )
                attention_block_names.append('cross_attention_vo')

        self.feed_forwards = nn.CellList([
            PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
            for _ in range(n_inputs)
        ])

        self.attention_block_name2idx = {
            name: idx for idx, name in enumerate(attention_block_names)
        }
        
        self.tile = ops.Tile()
        self.squeeze = ops.Squeeze(axis=-1)
        self.expand_dims = ops.ExpandDims()
        self.mul = ops.Mul()
        self.add = ops.Add()

    def _fetch_prefix(self, prefix_list, attn_name):
        if prefix_list is None:
            return None
        else:
            return prefix_list[self.attention_block_name2idx[attn_name]]

    def construct(
        self,
        x_tov,
        x_s,
        active_entries_treat_outcomes,
        active_entries_vitals=None,
        prefix_list=None,
    ):
        assert len(x_tov) == self.n_inputs
        
        if self.n_inputs == 2:
            x_t, x_o = x_tov
            x_v = None
        else:
            x_t, x_o, x_v = x_tov
            
        bsz, ch_t, h_t, w_t = x_t.shape
        l_t = h_t * w_t

        active_mask_ot = active_entries_treat_outcomes
        mask_mat_unsq = self.mul(self.expand_dims(active_mask_ot, 1), self.expand_dims(active_mask_ot, 2))
        self_att_mask_ot = self.expand_dims(self.squeeze(mask_mat_unsq), 1)

        cross_att_mask_ot = self_att_mask_ot
        cross_att_mask_to = self_att_mask_ot

        x_t_ = self.self_attention_t(
            x_t, x_t, x_t, self_att_mask_ot, True,
            prefix=self._fetch_prefix(prefix_list, 'self_attention_t'),
        )

        if (
            not self.disable_cross_attention
            and self.isolate_subnetwork not in ('t', 'o')
        ):
            x_to_ = self.cross_attention_to(
                x_t_, x_o, x_o, cross_att_mask_to, True,
                prefix=self._fetch_prefix(prefix_list, 'cross_attention_to'),
            )
        else:
            x_to_ = x_t_

        x_o_ = self.self_attention_o(
            x_o, x_o, x_o, self_att_mask_ot, True,
            prefix=self._fetch_prefix(prefix_list, 'self_attention_o'),
        )

        if (
            not self.disable_cross_attention
            and self.isolate_subnetwork not in ('o', 't')
        ):
            x_ot_ = self.cross_attention_ot(
                x_o_, x_t, x_t, cross_att_mask_ot, True,
                prefix=self._fetch_prefix(prefix_list, 'cross_attention_ot'),
            )
        else:
            x_ot_ = x_o_

        if self.n_inputs == 2:

            x_s_ffn = ops.reshape(ops.transpose(x_s, (0, 2, 3, 1)), (bsz, l_t, ch_t))

            x_to_ffn = ops.reshape(ops.transpose(x_to_, (0, 2, 3, 1)), (bsz, l_t, ch_t))
            x_ot_ffn = ops.reshape(ops.transpose(x_ot_, (0, 2, 3, 1)), (bsz, l_t, ch_t))
            
            out_t = self.feed_forwards[0](self.add(x_to_ffn, x_s_ffn))
            out_o = self.feed_forwards[1](self.add(x_ot_ffn, x_s_ffn))

            return out_t, out_o

        else:
            bsz, ch_v, h_v, w_v = x_v.shape
            l_v = h_v * w_v

            active_mask_v = active_entries_vitals
            mask_mat_unsq_v = self.mul(self.expand_dims(active_mask_v, 1), self.expand_dims(active_mask_v, 2))
            self_att_mask_v = self.expand_dims(self.squeeze(mask_mat_unsq_v), 1)

            active_to = active_entries_treat_outcomes
            active_v = active_entries_vitals

            cross_mask_q_to = self.expand_dims(self.squeeze(active_to, -1), 1)
            cross_mask_k_v = self.expand_dims(active_v, 1)
            cross_att_mask_ot_v = self.expand_dims(self.mul(cross_mask_q_to, ops.squeeze(cross_mask_k_v, -1)), 1)

            cross_mask_q_v = self.expand_dims(self.squeeze(active_v, -1), 1)
            cross_mask_k_to = self.expand_dims(active_to, 1)
            cross_att_mask_v_ot = self.expand_dims(self.mul(cross_mask_q_v, ops.squeeze(cross_mask_k_to, -1)), 1) # (B, 1, L_v, L_to)

            x_tv_ = 0.0
            x_ov_ = 0.0
            if not self.disable_cross_attention and self.isolate_subnetwork not in ('t', 'v', 'o'):
                x_tv_ = self.cross_attention_tv(
                    x_t_, x_v, x_v, cross_att_mask_ot_v, True,
                    prefix=self._fetch_prefix(prefix_list, 'cross_attention_tv'),
                )
                x_ov_ = self.cross_attention_ov(
                    x_o_, x_v, x_v, cross_att_mask_ot_v, True,
                    prefix=self._fetch_prefix(prefix_list, 'cross_attention_ov'),
                )

            x_v_ = self.self_attention_v(
                x_v, x_v, x_v, self_att_mask_v, True,
                prefix=self._fetch_prefix(prefix_list, 'self_attention_v'),
            )

            x_vt_ = x_v_
            x_vo_ = 0.0
            if not self.disable_cross_attention and self.isolate_subnetwork not in ('v', 't', 'o'):
                x_vt_ = self.cross_attention_vt(
                    x_v_, x_t, x_t, cross_att_mask_v_ot, True,
                    prefix=self._fetch_prefix(prefix_list, 'cross_attention_vt'),
                )
                x_vo_ = self.cross_attention_vo(
                    x_v_, x_o, x_o, cross_att_mask_v_ot, True,
                    prefix=self._fetch_prefix(prefix_list, 'cross_attention_vo'),
                )

            x_s_ffn_t = ops.reshape(ops.transpose(x_s, (0, 2, 3, 1)), (bsz, l_t, ch_t))
            x_s_ffn_v = ops.reshape(ops.transpose(x_s, (0, 2, 3, 1)), (bsz, l_v, ch_v))

            x_to_ffn = ops.reshape(ops.transpose(x_to_, (0, 2, 3, 1)), (bsz, l_t, ch_t))
            x_ot_ffn = ops.reshape(ops.transpose(x_ot_, (0, 2, 3, 1)), (bsz, l_t, ch_t))
            x_tv_ffn = ops.reshape(ops.transpose(x_tv_, (0, 2, 3, 1)), (bsz, l_t, ch_t))
            x_ov_ffn = ops.reshape(ops.transpose(x_ov_, (0, 2, 3, 1)), (bsz, l_t, ch_t))
            x_vt_ffn = ops.reshape(ops.transpose(x_vt_, (0, 2, 3, 1)), (bsz, l_v, ch_v))
            x_vo_ffn = ops.reshape(ops.transpose(x_vo_, (0, 2, 3, 1)), (bsz, l_v, ch_v))
            
            out_t = self.feed_forwards[0](self.add(x_to_ffn, self.add(x_tv_ffn, x_s_ffn_t)))
            out_o = self.feed_forwards[1](self.add(x_ot_ffn, self.add(x_ov_ffn, x_s_ffn_t)))
            out_v = self.feed_forwards[2](self.add(x_vt_ffn, self.add(x_vo_ffn, x_s_ffn_v)))

            return out_t, out_o, out_v


class TransformerSingleInputBlock(nn.Cell):
    """Transformer block with single input (MindSpore)."""

    def __init__(
        self,
        hidden,
        attn_heads,
        head_size,
        feed_forward_hidden,
        dropout,
        attn_dropout,
        self_positional_encoding_k=None,
        self_positional_encoding_v=None,
        n_inputs=1,
        final_layer=False,
        disable_cross_attention=False,
        isolate_subnetwork='',
        **kwargs,
    ):
        super(TransformerSingleInputBlock, self).__init__()
        self.n_inputs = n_inputs
        self.disable_cross_attention = disable_cross_attention
        self.isolate_subnetwork = isolate_subnetwork

        attention_block_names = []

        self.self_attention = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k,
            positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )
        attention_block_names.append('self_attention')

        self.feed_forwards = nn.CellList([
            PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
            for _ in range(n_inputs)
        ])

        self.attention_block_name2idx = {
            name: idx for idx, name in enumerate(attention_block_names)
        }
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze(axis=-1)
        self.mul = ops.Mul()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.add = ops.Add()

    def _fetch_prefix(self, prefix_list, attn_name):
        if prefix_list is None:
            return None
        else:
            return prefix_list[self.attention_block_name2idx[attn_name]]

    def construct(
        self,
        x_tov,
        x_s,
        active_entries_treat_outcomes,
        active_entries_vitals=None,
        prefix_list=None,
    ):
        x_t = x_tov
        bsz, ch, h, w = x_t.shape
        l_t = h * w

        active_mask_t = active_entries_treat_outcomes
        mask_mat_unsq = self.mul(self.expand_dims(active_mask_t, 1), self.expand_dims(active_mask_t, 2))
        self_att_mask_ot = self.expand_dims(self.squeeze(mask_mat_unsq), 1)

        x_t_ = self.self_attention(
            x_t, x_t, x_t, self_att_mask_ot, True,
            prefix=self._fetch_prefix(prefix_list, 'self_attention'),
        )

        x_t_ffn = self.reshape(self.transpose(x_t_, (0, 2, 3, 1)), (bsz, l_t, ch))
        x_s_ffn = self.reshape(self.transpose(x_s, (0, 2, 3, 1)), (bsz, l_t, ch))

        out_t = self.feed_forwards[0](self.add(x_t_ffn, x_s_ffn))
        
        return out_t