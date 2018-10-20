# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
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
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector
from nltk.translate.bleu_score import sentence_bleu
#### yaserkl@vt.edu adding rouge library
from rouge import rouge
from rouge_tensor import rouge_l_fscore
import data
from scipy.sparse import lil_matrix
from dqn import DQN
from replay_buffer import ReplayBuffer
from replay_buffer import Transition
from sklearn.preprocessing import OneHotEncoder

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def reward_function(self, summary, reference):
    """Calculate the reward between the reference and summary.

    Args:
      reference: A list of ids representing the ground-truth data
      summary: A list of ids representing the model generated data

    Returns:
      A single value representing the evaluation value for reference and summary
    """
    if 'rouge' in self._hps.reward_function:
      return rouge([summary],[reference])[self._hps.reward_function]
    else:
      return sentence_bleu([reference.split()],summary.split(),weights=(0.25,0.25,0.25,0.25))

  def variable_summaries(self, var_name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries_{}'.format(var_name)):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def _add_placeholders_common(self):
    """Add common placeholders to the graph. These are entry points for any input data."""
    hps = self._hps
    self._eta = tf.placeholder(tf.float32, None, name='eta')
    self._zeta = tf.placeholder(tf.float32, None, name='zeta')
    if FLAGS.embedding:
      self.embedding_place = tf.placeholder(tf.float32, [self._vocab.size(), hps.emb_dim])
    if FLAGS.scheduled_sampling:
      self._sampling_probability = tf.placeholder(tf.float32, None, name='sampling_probability')
      self._alpha = tf.placeholder(tf.float32, None, name='alpha')

  def _add_placeholders(self):
    """Add placeholders for full dataset to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
    if FLAGS.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

    # decoder part
    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

    if hps.mode == "decode":
      if hps.coverage:
        self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')
      if hps.intradecoder:
        self.prev_decoder_outputs = tf.placeholder(tf.float32, [None, hps.batch_size, hps.dec_hidden_dim], name='prev_decoder_outputs')
      if hps.use_temporal_attention:
        self.prev_encoder_es = tf.placeholder(tf.float32, [None, hps.batch_size, None], name='prev_encoder_es')


  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._enc_batch] = batch.enc_batch
    feed_dict[self._enc_lens] = batch.enc_lens
    feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    return feed_dict

  def _add_encoder(self, emb_enc_inputs, seq_len, partial=False):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      emb_enc_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of emb_enc_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder') as scope:
      #if self._hps.rl_training and partial:
      #  scope.reuse_variables()
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.enc_hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.enc_hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, emb_enc_inputs,
                                                                          dtype=tf.float32, sequence_length=seq_len,
                                                                          swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st

  def _reduce_states(self, fw_st, bw_st, partial=False):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    enc_hidden_dim = self._hps.enc_hidden_dim
    dec_hidden_dim = self._hps.dec_hidden_dim

    with tf.variable_scope('reduce_final_st') as scope:
      #if self._hps.rl_training and partial:
      #  scope.reuse_variables()
      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [enc_hidden_dim * 2, dec_hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [enc_hidden_dim * 2, dec_hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [dec_hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [dec_hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state

  def _add_decoder(self, emb_dec_inputs, embedding):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      emb_dec_inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)
      embedding: embedding matrix (vocab_size, emb_dim)
    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of tensors shape (batch_size, 1); the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    cell = tf.contrib.rnn.LSTMCell(hps.dec_hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

    prev_coverage = self.prev_coverage if (hps.mode=="decode" and hps.coverage) else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
    prev_decoder_outputs = self.prev_decoder_outputs if (hps.intradecoder and hps.mode=="decode") else tf.stack([],axis=0)
    prev_encoder_es = self.prev_encoder_es if (hps.use_temporal_attention and hps.mode=="decode") else tf.stack([],axis=0)
    return attention_decoder(_hps=hps,
      v_size=self._vocab.size(),
      _max_art_oovs=self._max_art_oovs,
      _enc_batch_extend_vocab=self._enc_batch_extend_vocab,
      emb_dec_inputs=emb_dec_inputs,
      target_batch=self._target_batch,
      _dec_in_state=self._dec_in_state,
      _enc_states=self._enc_states,
      enc_padding_mask=self._enc_padding_mask,
      dec_padding_mask=self._dec_padding_mask,
      cell=cell,
      embedding=embedding,
      sampling_probability=self._sampling_probability if FLAGS.scheduled_sampling else 0,
      alpha=self._alpha if FLAGS.E2EBackProp else 0,
      unk_id=self._vocab.word2id(data.UNKNOWN_TOKEN),
      initial_state_attention=(hps.mode=="decode"),
      pointer_gen=hps.pointer_gen,
      use_coverage=hps.coverage,
      prev_coverage=prev_coverage,
      prev_decoder_outputs=prev_decoder_outputs,
      prev_encoder_es = prev_encoder_es)

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  def discount_rewards(self, r):
    """ take a list of size max_dec_step * (batch_size, k) and return a list of the same size """
    discounted_r = []
    running_add = tf.constant(0, tf.float32)
    for t in reversed(range(0, len(r))):
      running_add = running_add * self._hps.gamma + r[t] # rd_t = r_t + gamma * r_{t+1}
      discounted_r.append(running_add)
    discounted_r = tf.stack(discounted_r[::-1]) # (max_dec_step, batch_size, k)
    normalized_discounted_r = tf.nn.l2_normalize(discounted_r, axis=0)
    return tf.unstack(normalized_discounted_r) # list of max_dec_step * (batch_size, k)

  def intermediate_rewards(self, r):
    """ take a list of size max_dec_step * (batch_size, k) and return a list of the same size
        uses the intermediate reward as proposed by: R_t = r_t - r_{t-1} """
    intermediate_r = []
    intermediate_r.append(r[0])
    for t in range(1, len(r)):
      intermediate_r.append(r[t]-r[t-1])
    return intermediate_r # list of max_dec_step * (batch_size, k)


  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        if FLAGS.embedding:
          embedding = tf.Variable(self.embedding_place)
        else:
          embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        if hps.mode=="train": self._add_emb_vis(embedding) # add to tensorboard
        emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)

      # Add the encoder.
      enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
      self._enc_states = enc_outputs

      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      self._dec_in_state = self._reduce_states(fw_st, bw_st)
      _t = time.time()
      # Add the decoder.
      with tf.variable_scope('decoder') as scope:
        (
        self.decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage,
        self.vocab_scores, self.final_dists, self.samples, self.greedy_search_samples, self.temporal_es,
        self.sampling_rewards, self.greedy_rewards) = self._add_decoder(emb_dec_inputs, embedding)
      tf.logging.info('adding decoder took {:0.3f} seconds'.format(time.time()-_t))
      if hps.rl_training and hps.mode in ['train', 'eval']: # separate tensors
        self._hps = self._hps._replace(batch_size=(self._hps.batch_size/2)) # separate the full/partial batches in this mini-batch
        hps = hps._replace(batch_size = hps.batch_size/2)
        self.decoder_outputs_full = tf.unstack(self.decoder_outputs[0:hps.batch_size], axis=1)
        self.decoder_outputs_partial = tf.unstack(self.decoder_outputs[hps.batch_size:], axis=1)
        self._dec_out_state_full = self._dec_out_state[0:hps.batch_size]
        self._dec_out_state_partial = self._dec_out_state[hps.batch_size:]
        self.attn_dists_full = tf.unstack(self.attn_dists[0:hps.batch_size], axis=1)
        self.attn_dists_partial = tf.unstack(self.attn_dists[hps.batch_size:], axis=1)
        self.p_gens_full = tf.unstack(self.p_gens[0:hps.batch_size], axis=1)
        self.p_gens_partial = tf.unstack(self.p_gens[hps.batch_size:], axis=1)
        self.coverage_full = self.coverage[0:hps.batch_size] if self.coverage is not None else None
        self.coverage_partial = self.coverage[hps.batch_size:] if self.coverage is not None else None
        self.vocab_scores_full = tf.unstack(self.vocab_scores[0:hps.batch_size], axis=1)
        self.vocab_scores_partial = tf.unstack(self.vocab_scores[hps.batch_size:], axis=1)
        self.final_dists_full = tf.unstack(self.final_dists[0:hps.batch_size], axis=1)
        self.final_dists_partial = tf.unstack(self.final_dists[hps.batch_size:], axis=1)
        self.samples_full = tf.unstack(self.samples[0:hps.batch_size], axis=1)
        self.samples_partial = tf.unstack(self.samples[hps.batch_size:], axis=1)
        self.greedy_search_samples_full = tf.unstack(self.greedy_search_samples[0:hps.batch_size], axis=1)
        self.greedy_search_samples_partial = tf.unstack(self.greedy_search_samples[hps.batch_size:], axis=1)
        self.temporal_es_full = tf.unstack(self.temporal_es[0:hps.batch_size], axis=1)
        self.temporal_es_partial = tf.unstack(self.temporal_es[hps.batch_size:], axis=1)
        self.sampling_rewards_full = tf.unstack(self.sampling_rewards[0:hps.batch_size], axis=1) \
          if (FLAGS.use_intermediate_rewards or FLAGS.use_discounted_rewards) else self.sampling_rewards[0:hps.batch_size]
        self.sampling_rewards_partial = tf.unstack(self.sampling_rewards[hps.batch_size:], axis=1) \
          if (FLAGS.use_intermediate_rewards or FLAGS.use_discounted_rewards) else self.sampling_rewards[hps.batch_size:]
        self.greedy_rewards_full = tf.unstack(self.greedy_rewards[0:hps.batch_size], axis=1) \
          if (FLAGS.use_intermediate_rewards or FLAGS.use_discounted_rewards) else self.greedy_rewards[0:hps.batch_size]
        self.greedy_rewards_partial = tf.unstack(self.greedy_rewards[hps.batch_size:], axis=1) \
          if (FLAGS.use_intermediate_rewards or FLAGS.use_discounted_rewards) else self.greedy_rewards[hps.batch_size:]

      if FLAGS.use_discounted_rewards and hps.rl_training and hps.mode in ['train', 'eval']:
        # Get the sampled and greedy sentence from model output
        # self.samples: (max_dec_steps, batch_size, k)
        self.sampling_discounted_rewards_full = tf.stack(
          self.discount_rewards(tf.unstack(self.sampling_rewards_full)))  # list of max_dec_steps * (batch_size, k)
        self.greedy_discounted_rewards_full = tf.stack(
          self.discount_rewards(tf.unstack(self.greedy_rewards_full)))  # list of max_dec_steps * (batch_size, k)
        self.sampling_discounted_rewards_partial = tf.stack(
          self.discount_rewards(tf.unstack(self.sampling_rewards_partial)))  # list of max_dec_steps * (batch_size, k)
        self.greedy_discounted_rewards_partial = tf.stack(
          self.discount_rewards(tf.unstack(self.greedy_rewards_partial)))  # list of max_dec_steps * (batch_size, k)

      elif FLAGS.use_intermediate_rewards and hps.rl_training and hps.mode in ['train', 'eval']:
        # Get the sampled and greedy sentence from model output
        # self.samples: (max_dec_steps, batch_size, k)
        self.sampling_discounted_rewards_full = tf.stack(
          self.intermediate_rewards(tf.unstack(self.sampling_rewards_full)))  # list of max_dec_steps * (batch_size, k)
        self.greedy_discounted_rewards_full = tf.stack(
          self.intermediate_rewards(tf.unstack(self.greedy_rewards_full)))  # list of max_dec_steps * (batch_size, k)
        self.sampling_discounted_rewards_partial = tf.stack(self.intermediate_rewards(
          tf.unstack(self.sampling_rewards_partial)))  # list of max_dec_steps * (batch_size, k)
        self.greedy_discounted_rewards_partial = tf.stack(
          self.intermediate_rewards(tf.unstack(self.greedy_rewards_partial)))  # list of max_dec_steps * (batch_size, k)

    if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
      self.decoder_outputs = tf.unstack(self.decoder_outputs, axis=1)
      self.attn_dists = tf.unstack(self.attn_dists, axis=1)
      self.p_gens = tf.unstack(self.p_gens, axis=1)
      self.vocab_scores = tf.unstack(self.vocab_scores, axis=1)
      self.final_dists = tf.unstack(self.final_dists, axis=1)
      self.samples = tf.unstack(self.samples, axis=1)
      self.greedy_search_samples = tf.unstack(self.greedy_search_samples, axis=1)
      self.temporal_es = tf.unstack(self.temporal_es, axis=1)
      self.sampling_rewards = tf.unstack(self.sampling_rewards, axis=1)
      self.greedy_rewards = tf.unstack(self.greedy_rewards, axis=1)
      assert len(self.final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
      self.final_dists = self.final_dists[0]
      topk_probs, self._topk_ids = tf.nn.top_k(self.final_dists, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
      self._topk_log_probs = tf.log(topk_probs)
      tf.logging.info('adding seq2seq took {:0.3f} seconds'.format(time.time()-_t))

  def _add_shared_loss_op(self):
    # Calculate the loss
    _t = time.time()
    with tf.variable_scope('shared_loss'):
      # Calculate the loss per step
      # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
      #### added by yaserkl@vt.edu: we just calculate these to monitor pgen_loss throughout time
      loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
      batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
      for dec_step, dist in enumerate(self.final_dists_full):
        if self._hps.rl_training:
          targets = self._target_batch[0:self._hps.batch_size, dec_step] # The indices of the target words. shape (batch_size)
        else:
          targets = self._target_batch[:, dec_step]  # The indices of the target words. shape (batch_size)
        indices = tf.stack((batch_nums, targets), axis=1) # shape (batch_size, 2)
        gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
        losses = -tf.log(gold_probs)
        loss_per_step.append(losses)
      if self._hps.rl_training:
        self._pgen_loss = _mask_and_avg(loss_per_step, self._dec_padding_mask[0:self._hps.batch_size])
      else:
        self._pgen_loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)
      self.variable_summaries('pgen_loss', self._pgen_loss)
      # Adding Self-Critic Reward to CE loss in Policy-Gradient Model
      #### Calculating the reinforce loss according to Eq. 15 in https://arxiv.org/pdf/1705.04304.pdf
      self._sampled_rouges_full = []
      self._greedy_rouges_full = []
      self._reward_diff_full = []
      for _ in range(self._hps.k):
        if FLAGS.use_discounted_rewards or FLAGS.use_intermediate_rewards:
          self._sampled_rouges_full.append(self.sampling_discounted_rewards_full[:, :, _])  # shape (max_enc_steps, batch_size)
          self._greedy_rouges_full.append(self.greedy_discounted_rewards_full[:, :, _])  # shape (max_enc_steps, batch_size)
        else:
          # use the reward of last step, since we use the reward of the whole sentence in this case
          self._sampled_rouges_full.append(self.sampling_rewards_full[:, _])  # shape (batch_size)
          self._greedy_rouges_full.append(self.greedy_rewards_full[:, _])  # shape (batch_size)
        if FLAGS.self_critic:
          self._reward_diff_full.append(self._greedy_rouges_full[_] - self._sampled_rouges_full[_]) # shape (max_enc_steps, batch_size)
        else:
          self._reward_diff_full.append(self._sampled_rouges_full[_]) # shape (batch_size)
      if self._hps.rl_training:
        self._sampled_rouges_partial = []
        self._greedy_rouges_partial = []
        self._reward_diff_partial = []
        for _ in range(self._hps.k):
          if FLAGS.use_discounted_rewards or FLAGS.use_intermediate_rewards:
            self._sampled_rouges_partial.append(
              self.sampling_discounted_rewards_partial[:, :, _])  # shape (max_enc_steps, batch_size)
            self._greedy_rouges_partial.append(
              self.greedy_discounted_rewards_partial[:, :, _])  # shape (max_enc_steps, batch_size)
          else:
            # use the reward of last step, since we use the reward of the whole sentence in this case
            self._sampled_rouges_partial.append(self.sampling_rewards_partial[:, _])  # shape (batch_size)
            self._greedy_rouges_partial.append(self.greedy_rewards_partial[:, _])  # shape (batch_size)
          if FLAGS.self_critic:
            self._reward_diff_partial.append(
              self._greedy_rouges_partial[_] - self._sampled_rouges_partial[_])  # shape (max_enc_steps, batch_size)
          else:
            self._reward_diff_partial.append(self._sampled_rouges_partial[_])  # shape (batch_size)

      tf.logging.info('calculating full reward took {:0.3f} seconds'.format(time.time()-_t))
      if self._hps.rl_training:
        ############################################
        ### Calculating RL loss for full dataset ###
        ############################################
        loss_per_step = [] # will be list length max_dec_steps containing shape (k, batch_size)
        rl_loss_per_step = [] # will be list length max_dec_steps containing shape (k, batch_size)
        batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
        for dec_step, dist in enumerate(self.final_dists_full):
          _samples = self.samples_full[dec_step] # (batch_size, k)
          _lps = [] # list of size k containing shape (batch_size)
          _rl_lps = [] # list of size k containing shape (batch_size)
          for _k in range(self._hps.k):
            targets = tf.squeeze(_samples[:, _k]) # The indices of the sampled words. shape (batch_size)
            indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
            gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
            losses = -tf.log(gold_probs) # shape (batch_size)
            _lps.append(losses)
            # Equation 15 in https://arxiv.org/pdf/1705.04304.pdf
            # Equal reward for all tokens
            if FLAGS.use_discounted_rewards or FLAGS.use_intermediate_rewards:
              rl_losses = -tf.log(gold_probs) * self._reward_diff_full[_k][dec_step, :]  # positive values
            else:
              rl_losses = -tf.log(gold_probs) * self._reward_diff_full[_k] # positive values
            _rl_lps.append(rl_losses)
          loss_per_step.append(tf.stack(_lps)) # (k, batch_size)
          rl_loss_per_step.append(tf.stack(_rl_lps)) # (k, batch_size)
        # new size: (k, max_dec_steps, batch_size)
        rl_loss_per_step = tf.unstack(tf.transpose(tf.stack(rl_loss_per_step), perm=[1, 0, 2])) # (k, max_dec_step, batch_size)
        loss_per_step = tf.unstack(tf.transpose(tf.stack(loss_per_step), perm=[1, 0, 2])) # (k, max_dec_step, batch_size)
        self._rl_avg_logprobs_full = tf.reduce_mean(
          [_mask_and_avg(tf.unstack(_lps), self._dec_padding_mask[:self._hps.batch_size]) for _lps in
           tf.unstack(loss_per_step)])
        self._rl_loss_full = tf.reduce_mean(
          [_mask_and_avg(tf.unstack(_lps), self._dec_padding_mask[:self._hps.batch_size]) for _lps in
           tf.unstack(rl_loss_per_step)])

        ###############################################
        ### Calculating RL loss for partial dataset ###
        ###############################################
        loss_per_step = [] # will be list length max_dec_steps containing shape (k, batch_size)
        rl_loss_per_step = [] # will be list length max_dec_steps containing shape (k, batch_size)
        batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
        for dec_step, dist in enumerate(self.final_dists_partial):
          _samples = self.samples_partial[dec_step] # (batch_size, k)
          _lps = [] # list of size k containing shape (batch_size)
          _rl_lps = [] # list of size k containing shape (batch_size)
          for _k in range(self._hps.k):
            targets = tf.squeeze(_samples[:, _k]) # The indices of the sampled words. shape (batch_size)
            indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
            gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
            losses = -tf.log(gold_probs) # shape (batch_size)
            _lps.append(losses)
            # Equation 15 in https://arxiv.org/pdf/1705.04304.pdf
            # Equal reward for all tokens
            if FLAGS.use_discounted_rewards or FLAGS.use_intermediate_rewards:
              rl_losses = -tf.log(gold_probs) * self._reward_diff_partial[_k][dec_step, :]  # positive values
            else:
              rl_losses = -tf.log(gold_probs) * self._reward_diff_partial[_k] # positive values
            _rl_lps.append(rl_losses)
          loss_per_step.append(tf.stack(_lps)) # (k, batch_size)
          rl_loss_per_step.append(tf.stack(_rl_lps)) # (k, batch_size)
        # new size: (k, max_dec_steps, batch_size)
        rl_loss_per_step = tf.unstack(tf.transpose(tf.stack(rl_loss_per_step), perm=[1,0,2])) # (k, max_dec_step, batch_size)
        loss_per_step = tf.unstack(tf.transpose(tf.stack(loss_per_step), perm=[1, 0, 2])) # (k, max_dec_step, batch_size)
        self._rl_avg_logprobs_partial = tf.reduce_mean(
          [_mask_and_avg(tf.unstack(_lps), self._dec_padding_mask[self._hps.batch_size:]) for _lps in
           tf.unstack(loss_per_step)])
        self._rl_loss_partial = tf.reduce_mean(
          [_mask_and_avg(tf.unstack(_lps), self._dec_padding_mask[self._hps.batch_size:]) for _lps in
           tf.unstack(rl_loss_per_step)])

      if FLAGS.use_intermediate_rewards:
        self._sampled_rouges_full = tf.reduce_sum(self._sampled_rouges_full, axis=1)
        self._greedy_rouges_full = tf.reduce_sum(self._greedy_rouges_full, axis=1)
        self._reward_diff_full = tf.reduce_sum(self._reward_diff_full, axis=1)
        if self._hps.rl_training:
          self._sampled_rouges_partial = tf.reduce_sum(self._sampled_rouges_partial, axis=1)
          self._greedy_rouges_partial = tf.reduce_sum(self._greedy_rouges_partial, axis=1)
          self._reward_diff_partial = tf.reduce_sum(self._reward_diff_partial, axis=1)
      self._sampled_rouges_full = tf.reduce_mean(self._sampled_rouges_full)
      self._greedy_rouges_full = tf.reduce_mean(self._greedy_rouges_full)
      self._reward_diff_full = tf.reduce_mean(self._reward_diff_full)
      if self._hps.rl_training:
        self._sampled_rouges_partial = tf.reduce_mean(self._sampled_rouges_partial)
        self._greedy_rouges_partial = tf.reduce_mean(self._greedy_rouges_partial)
        self._reward_diff_partial = tf.reduce_mean(self._reward_diff_partial)

      if self._hps.coverage:
        with tf.variable_scope('coverage_loss'):
          self._coverage_loss_full = _coverage_loss(self.attn_dists_full if self._hps.rl_training else self.attn_dists,
                                                    self._dec_padding_mask[0:self._hps.batch_size] \
                                                    if self._hps.rl_training else self._dec_padding_mask)
          self.variable_summaries('coverage_loss', self._coverage_loss_full)
      tf.logging.info('calculating rl loss took {:0.3f} seconds'.format(time.time()-_t))

  def _add_transfer_loss(self):
      # We multiply the ROUGE difference of sampling vs greedy sentence to the loss of all tokens in the sequence
      # Eq. 16 in https://arxiv.org/pdf/1705.04304.pdf and Eq. 34 in https://arxiv.org/pdf/1805.09461.pdf
      if self._hps.pointer_gen and self._hps.coverage:
          self._pointer_cov_total_loss = self._pgen_loss + self._hps.cov_loss_wt * self._coverage_loss_full
          self.variable_summaries('pointer_coverage_loss', self._pointer_cov_total_loss)
      if self._hps.rl_training:
        self._reinforce_shared_loss = (tf.constant(1., dtype=tf.float32) - self._eta) * self._pgen_loss + \
                                      self._eta * ((tf.constant(1., dtype=tf.float32) - self._zeta) *
                                                   self._rl_loss_full + self._zeta * self._rl_loss_partial)
        with tf.variable_scope('reinforce_loss'):
          #### the following is only for monitoring purposes
          self.variable_summaries('full_rl_loss', self._rl_loss_full)
          self.variable_summaries('full_rl_avg_logprobs', self._rl_avg_logprobs_full)
          self.variable_summaries('full_sampled_sent_reward', self._sampled_rouges_full)
          self.variable_summaries('full_greedy_sent_reward', self._greedy_rouges_full)
          self.variable_summaries('full_reward_diff', self._reward_diff_full)
          self.variable_summaries('partial_rl_loss', self._rl_loss_partial)
          self.variable_summaries('partial_rl_avg_logprobs', self._rl_avg_logprobs_partial)
          self.variable_summaries('partial_sampled_sent_reward', self._sampled_rouges_partial)
          self.variable_summaries('partial_greedy_sent_reward', self._greedy_rouges_partial)
          self.variable_summaries('partial_reward_diff', self._reward_diff_partial)
          self.variable_summaries('rl_shared_loss', self._reinforce_shared_loss)

          if self._hps.coverage:
            # Calculate coverage loss from the attention distributions
            self._reinforce_cov_total_loss = self._reinforce_shared_loss + self._hps.cov_loss_wt * self._coverage_loss_full
            self.variable_summaries('reinforce_coverage_loss', self._reinforce_cov_total_loss)

  def _add_shared_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._pgen_loss
    if self._hps.coverage:
      loss_to_minimize = self._pointer_cov_total_loss
    if self._hps.rl_training:
      loss_to_minimize = self._reinforce_shared_loss
      if self._hps.coverage:
        loss_to_minimize = self._reinforce_cov_total_loss

    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    self.epoch = (self.global_step * FLAGS.batch_size) / FLAGS.train_size
    new_lr = tf.cond(tf.greater(self.epoch, 0), lambda: self._hps.lr/tf.cast(self.epoch, tf.float32), lambda: self._hps.lr)
    #new_lr = self._hps.lr/self.epoch if self.epoch>0 else self._hps.lr
    optimizer = tf.train.AdagradOptimizer(new_lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    #optimizer = tf.train.AdamOptimizer()
    self._shared_train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._add_placeholders_common()
    #self._add_placeholders_full()
    #if self._hps.rl_training:
    #  self._add_placeholders_partial()
    self._add_placeholders()
    with tf.device("/device:GPU:{}".format(self._hps.gpu_num)):
      self._add_seq2seq()
      if self._hps.mode in ['train', 'eval']:
        self._add_shared_loss_op()
        self._add_transfer_loss()
      if self._hps.mode == 'train':
        self._add_shared_train_op()
      self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_steps(self, sess, batch, step):
    """ Run train steps
    Args:
      sess: seq2seq session
      batch: current batch
      step: training step
    """
    feed_dict = {}
    if self._hps.rl_training:
      if self._hps.fixed_eta:
        feed_dict[self._eta] = self._hps.eta
      else:
        feed_dict[self._eta] = min(step * self._hps.eta, 1.)
      if self._hps.fixed_zeta:
        feed_dict[self._zeta] = self._hps.zeta
      else:
        total_steps = 1./self._hps.eta
        _zeta = float(step - self._hps.rl_start_step)/(total_steps - self._hps.rl_start_step)
        _zeta = np.clip(_zeta, 0, self._hps.zeta_clipping)
        feed_dict[self._zeta] = _zeta
        tf.logging.info("zeta: {}".format(_zeta))

    if self._hps.scheduled_sampling:
      if self._hps.fixed_sampling_probability:
        feed_dict[self._sampling_probability] = self._hps.sampling_probability
      else:
        feed_dict[self._sampling_probability] = min(step * self._hps.sampling_probability,1.) # linear decay function
      ranges = [np.exp(float(step) * self._hps.alpha),np.finfo(np.float64).max] # to avoid overflow
      feed_dict[self._alpha] = np.log(ranges[np.argmin(ranges)]) # linear decay function

    feed_dict.update(self._make_feed_dict(batch))

    to_return = {'train_op': self._shared_train_op,
                 'summaries': self._summaries,
                 'pgen_loss': self._pgen_loss,
                 'global_step': self.global_step,
                 }
    to_return['full_ssr'] = self._sampled_rouges_full
    to_return['full_gsr'] = self._greedy_rouges_full
    to_return['full_reward_diff'] = self._reward_diff_full
    if self._hps.rl_training:
      to_return['full_rl_loss'] = self._rl_loss_full
      to_return['full_rl_avg_logprobs'] = self._rl_avg_logprobs_full
      to_return['partial_rl_loss']= self._rl_loss_partial
      to_return['partial_rl_avg_logprobs']= self._rl_avg_logprobs_partial
      to_return['partial_ssr']= self._sampled_rouges_partial
      to_return['partial_gsr']= self._greedy_rouges_partial
      to_return['partial_reward_diff'] = self._reward_diff_partial
      to_return['shared_loss']= self._reinforce_shared_loss

    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss_full
      if self._hps.pointer_gen:
        to_return['pointer_cov_total_loss'] = self._pointer_cov_total_loss
      if self._hps.rl_training:
        to_return['reinforce_cov_total_loss']= self._reinforce_cov_total_loss

    # We feed the collected reward and feed it back to model to update the loss
    try:
      return sess.run(to_return, feed_dict)
    except Exception as ex:
      tf.logging.info(ex.message)
      tf.logging.info(batch)

  def run_eval_steps(self, sess, batch, step):
    """ Run eval steps
    Args:
      sess: seq2seq session
      batch: current batch
      step: training step
      q_estimates = if using Actor-Critic model, this variable will feed
      the Q-estimates collected from Critic and use it to update the model
      loss
    """
    feed_dict = {}
    if self._hps.rl_training:
      if self._hps.fixed_eta:
        feed_dict[self._eta] = self._hps.eta
      else:
        feed_dict[self._eta] = min(step * self._hps.eta, 1.)
      if self._hps.fixed_zeta:
        feed_dict[self._zeta] = self._hps.zeta
      else:
        total_steps = 1./self._hps.eta
        _zeta = float(step - self._hps.rl_start_step)/(total_steps - self._hps.rl_start_step)
        feed_dict[self._zeta] = np.clip(_zeta, 0, self._hps.zeta_clipping)

    if self._hps.scheduled_sampling:
      if self._hps.fixed_sampling_probability:
        feed_dict[self._sampling_probability] = self._hps.sampling_probability
      else:
        feed_dict[self._sampling_probability] = min(step * self._hps.sampling_probability,1.) # linear decay function
      ranges = [np.exp(float(step) * self._hps.alpha),np.finfo(np.float64).max] # to avoid overflow
      feed_dict[self._alpha] = np.log(ranges[np.argmin(ranges)]) # linear decay function

    feed_dict.update(self._make_feed_dict(batch))

    to_return = {'summaries': self._summaries,
                 'pgen_loss': self._pgen_loss,
                 'global_step': self.global_step,
                 }
    to_return['full_ssr'] = self._sampled_rouges_full
    to_return['full_gsr'] = self._greedy_rouges_full
    to_return['full_reward_diff'] = self._reward_diff_full
    if self._hps.rl_training:
      to_return['full_rl_loss'] = self._rl_loss_full
      to_return['full_rl_avg_logprobs'] = self._rl_avg_logprobs_full
      to_return['partial_rl_loss']= self._rl_loss_partial
      to_return['partial_rl_avg_logprobs']= self._rl_avg_logprobs_partial
      to_return['partial_ssr']= self._sampled_rouges_partial
      to_return['partial_gsr']= self._greedy_rouges_partial
      to_return['partial_reward_diff'] = self._reward_diff_partial
      to_return['shared_loss']= self._reinforce_shared_loss

    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss_full
      if self._hps.pointer_gen:
        to_return['pointer_cov_total_loss'] = self._pointer_cov_total_loss
      if self._hps.rl_training:
        to_return['reinforce_cov_total_loss']= self._reinforce_cov_total_loss

    # We feed the collected reward and feed it back to model to update the loss
    try:
      return sess.run(to_return, feed_dict)
    except Exception as ex:
      tf.logging.info(ex.message)
      tf.logging.info(batch)

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, dec_in_state

  def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage, prev_decoder_outputs, prev_encoder_es):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    feed = {
        self._enc_states: enc_states,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
        self._dec_padding_mask: np.ones((beam_size,1),dtype=np.float32)
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self.attn_dists,
      "final_dists": self.final_dists
    }

    if FLAGS.pointer_gen:
      feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._max_art_oovs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens

    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage

    if FLAGS.ac_training or FLAGS.intradecoder:
      to_return['output']=self.decoder_outputs
    if FLAGS.intradecoder:
      feed[self.prev_decoder_outputs]= prev_decoder_outputs
    if FLAGS.use_temporal_attention:
      to_return['temporal_e'] = self.temporal_es
      feed[self.prev_encoder_es] = prev_encoder_es

    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
                  range(beam_size)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()
    final_dists = results['final_dists'][0].tolist()

    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens'])==1
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in range(beam_size)]

    if FLAGS.ac_training or FLAGS.intradecoder:
      output = results['output'][0] # used for calculating the intradecoder at later steps and for calcualting q-estimate in Actor-Critic training.
    else:
      output = None
    if FLAGS.use_temporal_attention:
      temporal_e = results['temporal_e'][0] # used for calculating the attention at later steps
    else:
      temporal_e = None

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in range(beam_size)]

    return results['ids'], results[
      'probs'], new_states, attn_dists, final_dists, p_gens, new_coverage, output, temporal_e


def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)] # list of k
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average

def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss
