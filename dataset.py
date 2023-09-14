import tensorflow as tf
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)

from tensorflow import keras
from keras.layers import Embedding,LayerNormalization,Dropout
from keras.layers import Dense,Embedding,Concatenate,Masking
from keras.models import Model,Sequential, load_model

class BilingualDataset(tf.keras.utils.Sequence):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = tf.constant([tokenizer_tgt.token_to_id("[SOS]")], dtype=tf.int64)
        self.eos_token = tf.constant([tokenizer_tgt.token_to_id("[EOS]")], dtype=tf.int64)
        self.pad_token = tf.constant([tokenizer_tgt.token_to_id("[PAD]")], dtype=tf.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        print(f"Fetching item at index {idx}")
        if idx >= len(self.ds):
          raise IndexError("Index out of bounds")
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos, and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s> and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = tf.concat(
            [
                self.sos_token,
                tf.constant(enc_input_tokens, dtype=tf.int64),
                self.eos_token,
                tf.constant([self.pad_token] * enc_num_padding_tokens, dtype=tf.int64),
            ],
            axis=0,
        )

        # Add only <s> token
        decoder_input = tf.concat(
            [
                self.sos_token,
                tf.constant(dec_input_tokens, dtype=tf.int64),
                tf.constant([self.pad_token] * dec_num_padding_tokens, dtype=tf.int64),
            ],
            axis=0,
        )

        # Add only </s> token
        label = tf.concat(
            [
                tf.constant(dec_input_tokens, dtype=tf.int64),
                self.eos_token,
                tf.constant([self.pad_token] * dec_num_padding_tokens, dtype=tf.int64),
            ],
            axis=0,
        )

        # Double-check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token)[:, tf.newaxis, tf.newaxis].numpy(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token)[:, tf.newaxis].numpy() & causal_mask(decoder_input.shape[0]),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = tf.linalg.band_part(tf.ones((1, size, size), dtype=tf.int64), -1, 0)
    return tf.math.equal(mask, 0)
