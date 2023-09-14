from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path
from model import build_transformer

from tensorflow import keras
from keras.layers import Embedding,LayerNormalization,Dropout,Dense
from keras.models import Model,Sequential, load_model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf



def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = tf.fill([1, 1], sos_idx)
    while True:
        if decoder_input.shape[1] == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.shape[1])

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        next_word = tf.argmax(prob, axis=1)
        decoder_input = tf.concat([decoder_input, tf.fill([1, 1], next_word.numpy()[0])], axis=1)

        if next_word == eos_idx:
            break

    return decoder_input[0, 1:]

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.compile()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    for batch in validation_ds:
        count += 1
        encoder_input = batch["encoder_input"] # (b, seq_len)
        encoder_mask = batch["encoder_mask"] # (b, 1, 1, seq_len)

        # check that the batch size is 1
        assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"

        model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        model_out_text = tokenizer_tgt.decode(np.array(model_out))

        source_texts.append(source_text)
        expected.append(target_text)
        predicted.append(model_out_text)

        # Print the source, target, and model output
        print_msg('-' * console_width)
        print_msg(f"{f'SOURCE: ':>12}{source_text}")
        print_msg(f"{f'TARGET: ':>12}{target_text}")
        print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

        if count == num_examples:
            print_msg('-' * console_width)
            break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        cer = tfa.metrics.CharacterErrorRate()
        cer.update_state(expected, predicted)
        writer.scalar('validation cer', cer.result().numpy(), step=global_step)
        writer.flush()

        # Compute the word error rate
        wer = tfa.metrics.WordErrorRate()
        wer.update_state(expected, predicted)
        writer.scalar('validation wer', wer.result().numpy(), step=global_step)
        writer.flush()

        # Compute the BLEU metric
        bleu = tf.py_function(tfa.metrics.bleu_score, (expected, predicted), tf.float32)
        writer.scalar('validation BLEU', bleu, step=global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it ourselves
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = train_test_split(ds_raw, test_size=val_ds_size, random_state=42)

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = tf.data.Dataset.from_generator(lambda: train_ds, output_signature=(
        {"encoder_input": tf.TensorSpec(shape=(None,), dtype=tf.int32),
         "decoder_input": tf.TensorSpec(shape=(None,), dtype=tf.int32),
         "encoder_mask": tf.TensorSpec(shape=(1, 1, None), dtype=tf.float32),
         "decoder_mask": tf.TensorSpec(shape=(1, None, None), dtype=tf.float32),
         "label": tf.TensorSpec(shape=(None,), dtype=tf.int32),
         "src_text": tf.TensorSpec(shape=(), dtype=tf.string),
         "tgt_text": tf.TensorSpec(shape=(), dtype=tf.string)})).batch(
        config['batch_size']).shuffle(1000, reshuffle_each_iteration=True)

    val_dataloader = tf.data.Dataset.from_generator(lambda: val_ds, output_signature=(
        {"encoder_input": tf.TensorSpec(shape=(None,), dtype=tf.int32),
         "decoder_input": tf.TensorSpec(shape=(None,), dtype=tf.int32),
         "encoder_mask": tf.TensorSpec(shape=(1, 1, None), dtype=tf.float32),
         "decoder_mask": tf.TensorSpec(shape=(1, None, None), dtype=tf.float32),
         "label": tf.TensorSpec(shape=(None,), dtype=tf.int32),
         "src_text": tf.TensorSpec(shape=(), dtype=tf.string),
         "tgt_text": tf.TensorSpec(shape=(), dtype=tf.string)})).batch(
        1).shuffle(1000, reshuffle_each_iteration=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model


def train_model(config):
    # Define the device
    device = "cuda" if tf.config.experimental.list_physical_devices("GPU") else "cpu"
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    optimizer = Adam(learning_rate=config['lr'], epsilon=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = tf.train.load_checkpoint(model_filename)
        model.set_weights(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.set_weights(state['optimizer_state_dict'])
        global_step = state['global_step']

    # Use CategoricalCrossentropy loss with label smoothing
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1, reduction=tf.keras.losses.Reduction.NONE)

    for epoch in range(initial_epoch, config['num_epochs']):
        tf.keras.backend.clear_session()
        model.compile(optimizer=optimizer)

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            with tf.GradientTape() as tape:  # Define the gradient tape
                encoder_input = batch['encoder_input'] # (b, seq_len)
                decoder_input = batch['decoder_input'] # (B, seq_len)
                encoder_mask = batch['encoder_mask'] # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'] # (B, 1, seq_len, seq_len)

                # Run the tensors through the encoder, decoder, and the projection layer
                encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
                proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

                # Compare the output with the label
                label = batch['label'] # (B, seq_len)

                # Compute the loss using the label smoothing loss function
                loss = loss_fn(label, proj_output)
                # Apply a mask to ignore padding tokens
                mask = tf.math.logical_not(tf.math.equal(label, tokenizer_tgt.token_to_id('[PAD]')))
                loss = tf.boolean_mask(loss, mask)
                # Calculate the mean loss
                loss = tf.reduce_mean(loss)

            batch_iterator.set_postfix({"loss": f"{loss.numpy():6.3f}"})

            # Log the loss
            writer = tf.summary.create_file_writer(config['experiment_name'])
            with writer.as_default():
                tf.summary.scalar('train loss', loss.numpy(), step=global_step)
                writer.flush()

            # Backpropagate the loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        model.save_weights(model_filename)
        checkpoint = tf.train.Checkpoint(
            epoch=tf.Variable(epoch),
            model_state_dict=model.get_weights(),
            optimizer_state_dict=optimizer.get_weights(),
            global_step=tf.Variable(global_step))
        checkpoint.save(model_filename)
if __name__ == '__main__':
    config = get_config()
    train_model(config)