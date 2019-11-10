import math

import torch
import torch.nn as nn
import torch.onnx.operators

from fairseq import utils


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False

    def forward(self, input, incremental_state=None, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (
            (positions is None) or (self.padding_idx is None)
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = input.data.new(1, 1).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = utils.make_positions(
                    input.data, self.padding_idx, onnx_trace=self.onnx_trace,
                )
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        if self.padding_idx is not None:
            return self.num_embeddings - self.padding_idx - 1
        else:
            return self.num_embeddings

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.onnx_trace = False
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(bsz, 1, 1)
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(input, self.padding_idx, onnx_trace=self.onnx_trace)
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat((bsz.view(1), seq_len.view(1), torch.LongTensor([-1])))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(flat_embeddings, embedding_shape)
            return embeddings
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


def PositionalEmbedding(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        learned: bool = False,
):
    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim, padding_idx, init_size=num_embeddings + padding_idx + 1,
        )
    return m


class RelativeSinusoidalPositionalEncoding(nn.Module):
    """This module produces relative sinusoidal positional embeddings.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, num_heads, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.max_size = init_size
        self.weights = RelativeSinusoidalPositionalEncoding.get_embedding(
            init_size,
            embedding_dim,
        )
        self.pos_vec = nn.Parameter(
            torch.Tensor(num_heads, embedding_dim))

        self.reset_parameters()
        self.onnx_trace = False

    def reset_parameters(self):
        std = math.sqrt(1.0 / float(self.embedding_dim))
        nn.init.normal_(self.pos_vec, std=std)

    @staticmethod
    def get_embedding(max_size, embedding_dim):
        """Build relative sinusoidal embeddings.
        """
        half_dim = embedding_dim // 2
        freq_seq = torch.arange(0, half_dim, 1.0, dtype=torch.float)
        inv_freq = 1 / (10000 ** (freq_seq / half_dim))

        pos_seq = torch.arange(max_size, -max_size, -1.0, dtype=torch.float)

        sinusoid_inp = torch.einsum("t,d->td", pos_seq, inv_freq)
        emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], -1)

        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(2 * max_size, 1)], dim=1)
        return emb

    @staticmethod
    def rel_shift(x, row_dim, key_len):
        """Perform relative shift to form the relative attention score."""
        x_size = x.size()

        # Deal with negative indexing
        if row_dim < 0:
            row_dim = x.ndim + row_dim
            assert row_dim >= 0

        # Assume `col_dim` = `row_dim + 1`
        col_dim = row_dim + 1
        assert col_dim < x.ndim

        tgt_shape_1, tgt_shape_2 = [], []
        for i in range(x.ndim):
            if i == row_dim:
                tgt_shape_1.append(x_size[col_dim])
                tgt_shape_2.append(x_size[row_dim])
            elif i == col_dim:
                tgt_shape_1.append(x_size[row_dim])
                tgt_shape_2.append(x_size[col_dim] - 1)
            else:
                tgt_shape_1.append(x_size[i])
                tgt_shape_2.append(x_size[i])

        x = x.reshape(tgt_shape_1)
        x = torch.narrow(x, row_dim, 1, x.size(row_dim) - 1)
        x = x.reshape(tgt_shape_2)
        x = torch.narrow(x, col_dim, 0, key_len)

        return x

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, input, incremental_state=None, timestep=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)
        if self.weights is None or seq_len > self.max_size:
            # recompute/expand embeddings if needed
            self.weights = RelativeSinusoidalPositionalEncoding.get_embedding(
                seq_len,
                self.embedding_dim,
            )
            self.max_size = seq_len
        self.weights = self.weights.to(self.pos_vec)

        if incremental_state is not None:
            pos_enc = self.weights[self.max_size-seq_len+1:self.max_size+1]
            pos_enc = pos_enc.detach()
            attn_bias = torch.einsum("nd,td->nt", self.pos_vec, pos_enc)
            attn_bias = attn_bias[:, None, :]
        else:
            pos_enc = self.weights[self.max_size-seq_len:self.max_size+seq_len]
            pos_enc = pos_enc.detach()
            attn_bias = torch.einsum("nd,td->nt", self.pos_vec, pos_enc)
            attn_bias = attn_bias[:, None, :].repeat(1, seq_len, 1)
            attn_bias = RelativeSinusoidalPositionalEncoding.rel_shift(
                 attn_bias, row_dim=-2, key_len=seq_len)

        return attn_bias

