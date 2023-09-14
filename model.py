import tensorflow as tf
import math
from tensorflow import keras
from keras.layers import Embedding,LayerNormalization,Dropout
from keras.layers import Dense
from keras.models import Model,Sequential, load_model

class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, eps:float=10**-6) -> None:
        super(LayerNormalization, self).__init__()
        self.eps = eps

    def build(self, input_shape):
        self.alpha = self.add_weight(name="alpha", shape=(1,), initializer="ones", trainable=True)
        self.bias = self.add_weight(name="bias", shape=(1,), initializer="zeros", trainable=True)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(tf.keras.layers.Layer):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = Dense(d_ff, activation="relu")
        self.dropout = Dropout(dropout)
        self.linear_2 = Dense(d_model)

    def call(self, x):
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class InputEmbeddings(tf.keras.layers.Layer):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(InputEmbeddings, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.d_model = d_model

    def call(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = Dropout(dropout)
        self.positional_encoding = self.build_positional_encoding(d_model, seq_len)

    def build_positional_encoding(self, d_model, seq_len):
        position = tf.range(0, seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(math.log(10000.0) / d_model))
        pe = tf.math.sin(position * div_term)
        pe = pe[tf.newaxis, :, :]  # Add batch dimension
        return pe

    def call(self, x):
        x = x + self.positional_encoding[:, :tf.shape(x)[1], :]
        return self.dropout(x)


class ResidualConnection(tf.keras.layers.Layer):

    def __init__(self, dropout: float) -> None:
        super(ResidualConnection, self).__init__()
        self.dropout = Dropout(dropout)
        self.norm = LayerNormalization()

    def call(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(tf.keras.layers.Layer):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = Dense(d_model, use_bias=False)
        self.w_k = Dense(d_model, use_bias=False)
        self.w_v = Dense(d_model, use_bias=False)
        self.w_o = Dense(d_model, use_bias=False)
        self.dropout = Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = tf.shape(query)[-1]
        attention_scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))
        if mask is not None:
            attention_scores = tf.where(mask == 0, tf.constant(-1e9, dtype=tf.float32), attention_scores)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        output = tf.matmul(attention_scores, value)
        return output, attention_scores

    def call(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = tf.reshape(query, (-1, tf.shape(query)[1], self.h, self.d_k))
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.reshape(key, (-1, tf.shape(key)[1], self.h, self.d_k))
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.reshape(value, (-1, tf.shape(value)[1], self.h, self.d_k))
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (-1, tf.shape(x)[1], self.d_model))
        return self.w_o(x)

class EncoderBlock(tf.keras.layers.Layer):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = [ResidualConnection(dropout) for _ in range(2)]

    def call(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(tf.keras.layers.Layer):

    def __init__(self, layers: list) -> None:
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def call(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = [ResidualConnection(dropout) for _ in range(3)]

    def call(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(tf.keras.layers.Layer):

    def __init__(self, layers: list) -> None:
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def call(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, vocab_size) -> None:
        super(ProjectionLayer, self).__init__()
        self.proj = Dense(vocab_size)

    def call(self, x):
        return tf.math.log(tf.nn.softmax(self.proj(x), axis=-1))

class Transformer(tf.keras.Model):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    return transformer
