"""Fairseq-based implementation of the model proposed in
   `"Joint Source-Target Self Attention with Locality Constraints" (Fonollosa, et al, 2019)
    <https://>`_.
   Author: Jose A. R. Fonollosa, Universitat Politecnica de Catalunya.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options
from fairseq import utils

from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel
)

from .protect_embeddings import(
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RelativeSinusoidalPositionalEncoding
)
from .protect_layers import(
    TransformerDecoderLayer
)
from .protect_modules import (
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention
)

@register_model('my_joint_attention')
class JointAttentionModel(FairseqEncoderDecoderModel):
    """
    Local Joint Source-Target model from
    `"Joint Source-Target Self Attention with Locality Constraints" (Fonollosa, et al, 2019)
    <https://>`_.

    Args:
        encoder (JointAttentionEncoder): the encoder
        decoder (JointAttentionDecoder): the decoder

    The joint source-target model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.joint_attention_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='embedding dimension for FFN')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--relative-attn', action='store_true', default=False,
                            help='relative attention')
        parser.add_argument('--softmax-bias', action='store_true', default=False,
                            help='softmax_bias')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    'The joint_attention model requires --encoder-embed-dim to match --decoder-embed-dim')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = JointAttentionEncoder(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        decoder = JointAttentionDecoder(args, tgt_dict, decoder_embed_tokens, left_pad=args.left_pad_target)
        return JointAttentionModel(encoder, decoder)


class JointAttentionEncoder(FairseqEncoder):
    """
    JointAttention encoder is used only to compute the source embeddings.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool): whether the input is left-padded
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)

        if args.no_token_positional_embeddings or args.relative_attn:
            self.embed_positions = None
        else:
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, embed_dim, self.padding_idx,
                learned=args.encoder_learned_pos,
            )
        self.embed_language = LanguageEmbedding(embed_dim)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): embedding output of shape
                  `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed layer
        x = self.embed_layer(src_tokens)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def embed_layer(self, x):
        # embed positions
        positions = self.embed_positions(
            x) if self.embed_positions is not None else None
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(x)
        if positions is not None:
            x += positions
        if self.embed_language is not None:
            lang_emb = self.embed_scale * self.embed_language.view(1, 1, -1)
            x += lang_emb
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class JointAttentionDecoder(FairseqIncrementalDecoder):
    """
    JointAttention decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)

        self.relative_attn = args.relative_attn
        if self.relative_attn:
            self.rel_attn_encoding = RelativeSinusoidalPositionalEncoding(
                embedding_dim=embed_dim,
                num_heads=args.decoder_attention_heads)

        if args.no_token_positional_embeddings or args.relative_attn:
            self.embed_positions = None
        else:
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, embed_dim, self.padding_idx,
                learned=args.decoder_learned_pos,
            )

        self.embed_language = LanguageEmbedding(embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn=True)
            for _ in range(args.decoder_layers)
        ])

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
        self.has_softmax_bias = args.softmax_bias
        if args.softmax_bias:
            self.softmax_bias = nn.Parameter(torch.Tensor(len(dictionary),))

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        """
        Args:
            input (dict): with
                prev_output_tokens (LongTensor): previous decoder outputs of shape
                    `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """

        tgt_len = prev_output_tokens.size(1)
        x = self.embed_layer(prev_output_tokens, incremental_state=incremental_state)

        # B x T x C -> T x B x C
        target = prev_output_tokens.transpose(0, 1)
        source = encoder_out['encoder_out']
        x = x.transpose(0, 1)

        bsz = target.size(1)
        seq_len = target.size(0) + source.size(0)
        tmp = torch.zeros(bsz, seq_len, device=target.device)
        # Compute relative attention bias
        self_attn_bias = self.rel_attn_encoding(
            tmp,
            incremental_state=incremental_state,
        ) if self.relative_attn else None

        # extended attention mask
        source_padding_mask = encoder_out['encoder_padding_mask']
        target_padding_mask = source_padding_mask.new_zeros((bsz, tgt_len))
        padding_mask = torch.cat((source_padding_mask, target_padding_mask), 1)
        if incremental_state is None:
            #no incremental forward, used in training and validation
            attn_mask = source.new_zeros(seq_len, seq_len)
            target_mask = self.buffered_future_mask(target)
            attn_mask[-tgt_len:, -tgt_len:] = target_mask
            src_to_tgt_mask = utils.fill_with_neg_inf(source.new(source.size(0), tgt_len))
            attn_mask[:-tgt_len, -tgt_len:] = src_to_tgt_mask
            x = torch.cat([source, x], axis=0)
            x = self.tfm_forward(x,
                attn_mask=attn_mask,
                padding_mask=padding_mask,
                self_attn_bias=self_attn_bias)
            x = x[-tgt_len:]
        else:
            #inference phase
            if len(incremental_state) == 0:
                #process source sentence
                tmp = torch.zeros(bsz, source.size(0), device=target.device)
                source_attn_bias = self.rel_attn_encoding(
                    tmp) if self.relative_attn else None
                source = self.tfm_forward(source,
                    padding_mask=source_padding_mask,
                    self_attn_bias=source_attn_bias,
                    incremental_state=incremental_state)
            key_padding_mask = prev_output_tokens[:, -1:].eq(self.padding_idx)
            x = self.tfm_forward(x,
                padding_mask=key_padding_mask,
                incremental_state=incremental_state,
                self_attn_bias=self_attn_bias)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        pred = self.output_layer(x)
        info = {}
        return pred, info

    def embed_layer(self, x, incremental_state):
        # embed positions
        positions = self.embed_positions(
            x,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None
        # use the last-step prediction as the next-step input
        if incremental_state is not None:
            x = x[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(x)
        if positions is not None:
            x += positions
        if self.embed_language is not None:
            lang_emb = self.embed_scale * self.embed_language.view(1, 1, -1)
            x += lang_emb
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def tfm_forward(
        self,
        x,
        attn_mask=None,
        padding_mask=None,
        incremental_state=None,
        self_attn_bias=None):
        for i, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_mask=attn_mask,
                self_attn_padding_mask=padding_mask,
                incremental_state=incremental_state,
                self_attn_bias=self_attn_bias)
        return x

    def output_layer(self, x):
        # project back to size of vocabulary
        if self.share_input_output_embed:
            if self.has_softmax_bias:
                x = F.linear(x, self.embed_tokens.weight, self.softmax_bias)
            else:
                x = F.linear(x, self.embed_tokens.weight)
        else:
            if self.has_sofmtax_bias:
                x = F.linear(x, self.embed_out, self.softmax_bias)
            else:
                x = F.linear(x, self.embed_out)
        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        """Cached future mask."""
        dim = tensor.size(0)
        #pylint: disable=access-member-before-definition, attribute-defined-outside-init
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LanguageEmbedding(embedding_dim):
    m = nn.Parameter(torch.Tensor(embedding_dim))
    nn.init.normal_(m, mean=0, std=embedding_dim ** -0.5)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('my_joint_attention', 'my_joint_attention')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)

    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_layers = getattr(args, 'decoder_layers', 14)

    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)

@register_model_architecture('my_joint_attention', 'my_joint_attention_wmt_en_de_big')
def joint_attention_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)
