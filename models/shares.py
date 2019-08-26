import os
import codecs
import json
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from utils.logger import get_logger
from utils.CoNLLeval import CoNLLeval
from sklearn.manifold import TSNE


class FlipGradientBuilder:
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, lw=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * lw]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()


class CharTDNNHW:
    def __init__(self, kernels, kernel_features, dim, hw_layers, padding="VALID", activation=tf.nn.relu, use_bias=True,
                 hw_activation=tf.nn.tanh, reuse=None, scope="char_tdnn_hw"):
        assert len(kernels) == len(kernel_features), "kernel and features must have the same size"
        self.padding, self.activation, self.reuse, self.scope = padding, activation, reuse, scope
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.weights = []
            for i, (kernel_size, feature_size) in enumerate(zip(kernels, kernel_features)):
                weight = tf.get_variable("filter_%d" % i, shape=[1, kernel_size, dim, feature_size], dtype=tf.float32)
                bias = tf.get_variable("bias_%d" % i, shape=[feature_size], dtype=tf.float32)
                self.weights.append((weight, bias))
            self.dense_layers = []
            for i in range(hw_layers):
                trans = tf.layers.Dense(units=sum(kernel_features), use_bias=use_bias, activation=hw_activation,
                                        name="trans_%d" % i)
                gate = tf.layers.Dense(units=sum(kernel_features), use_bias=use_bias, activation=tf.nn.sigmoid,
                                       name="gate_%d" % i)
                self.dense_layers.append((trans, gate))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            # cnn
            strides = [1, 1, 1, 1]
            outputs = []
            for i, (weight, bias) in enumerate(self.weights):
                conv = tf.nn.conv2d(inputs, weight, strides=strides, padding=self.padding, name="conv_%d" % i) + bias
                output = tf.reduce_max(self.activation(conv), axis=2)
                outputs.append(output)
            outputs = tf.concat(values=outputs, axis=-1)
            # highway
            for trans, gate in self.dense_layers:
                g = gate(outputs)
                outputs = g * trans(outputs) + (1.0 - g) * outputs
            return outputs


class CharTDNN:
    def __init__(self, kernels, kernel_features, dim, padding="VALID", activation=tf.nn.relu, reuse=None,
                 scope="char_tdnn"):
        assert len(kernels) == len(kernel_features), "kernel and features must have the same size"
        self.padding, self.activation, self.reuse, self.scope = padding, activation, reuse, scope
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.weights = []
            for i, (kernel_size, feature_size) in enumerate(zip(kernels, kernel_features)):
                weight = tf.get_variable("filter_%d" % i, shape=[1, kernel_size, dim, feature_size], dtype=tf.float32)
                bias = tf.get_variable("bias_%d" % i, shape=[feature_size], dtype=tf.float32)
                self.weights.append((weight, bias))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            strides = [1, 1, 1, 1]
            outputs = []
            for i, (weight, bias) in enumerate(self.weights):
                conv = tf.nn.conv2d(inputs, weight, strides=strides, padding=self.padding, name="conv_%d" % i) + bias
                output = tf.reduce_max(self.activation(conv), axis=2)
                outputs.append(output)
            outputs = tf.concat(values=outputs, axis=-1)
            return outputs


class BiRNN:
    def __init__(self, num_units, drop_rate=0.0, activation=tf.tanh, concat=True, reuse=None, scope="bi_rnn"):
        self.reuse, self.scope, self.concat = reuse, scope, concat
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
            self.cell_bw = tf.nn.rnn_cell.LSTMCell(num_units)
            self.rnn_dropout = tf.layers.Dropout(rate=drop_rate)
            if not self.concat:
                self.dense_fw = tf.layers.Dense(units=num_units, use_bias=False, _reuse=self.reuse, name="dense_fw")
                self.dense_bw = tf.layers.Dense(units=num_units, use_bias=False, _reuse=self.reuse, name="dense_bw")
                self.bias = tf.get_variable(name="bias", shape=[num_units], dtype=tf.float32, trainable=True)
                self.activation = activation

    def __call__(self, inputs, seq_len, training):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs, seq_len, dtype=tf.float32)
            if self.concat:
                outputs = tf.concat(outputs, axis=-1)
                outputs = self.rnn_dropout(outputs, training=training)
            else:
                output1 = self.rnn_dropout(outputs[0], training=training)
                output2 = self.rnn_dropout(outputs[1], training=training)
                outputs = self.dense_fw(output1) + self.dense_bw(output2)
                outputs = self.activation(tf.nn.bias_add(outputs, bias=self.bias))
            return outputs


class HighwayNetwork:
    def __init__(self, layers, num_units, activation=tf.tanh, use_bias=True, reuse=None, scope="highway"):
        self.reuse, self.scope = reuse, scope
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.dense_layers = []
            for i in range(layers):
                trans = tf.layers.Dense(units=num_units, use_bias=use_bias, activation=activation, name="trans_%d" % i)
                gate = tf.layers.Dense(units=num_units, use_bias=use_bias, activation=tf.nn.sigmoid, name="gate_%d" % i)
                self.dense_layers.append((trans, gate))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            for trans, gate in self.dense_layers:
                g = gate(inputs)
                inputs = g * trans(inputs) + (1.0 - g) * inputs
            return inputs


class CRF:
    def __init__(self, num_units, reuse=None, scope="crf"):
        self.reuse, self.scope = reuse, scope
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.transition = tf.get_variable(name="transition", shape=[num_units, num_units], dtype=tf.float32)

    def __call__(self, inputs, labels, seq_len):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            crf_loss, transition = tf.contrib.crf.crf_log_likelihood(inputs, labels, seq_len, self.transition)
            return transition, tf.reduce_mean(-crf_loss)


class Embedding:
    def __init__(self, token_size, token_dim, token2vec=None, token_weight=None, tune_emb=True, at=False, norm_emb=True,
                 word_project=False, scope="word_table"):
        self.scope, self.word_project, self.at, self.norm_emb = scope, word_project, at, norm_emb
        with tf.variable_scope(self.scope):
            if token2vec is not None:
                table = tf.Variable(initial_value=np.load(token2vec)["embeddings"], name="table", dtype=tf.float32,
                                    trainable=tune_emb)
                unk = tf.get_variable(name="unk", shape=[1, token_dim], trainable=True, dtype=tf.float32)
                table = tf.concat([unk, table], axis=0)
            else:
                table = tf.get_variable(name="table", shape=[token_size - 1, token_dim], dtype=tf.float32,
                                        trainable=True)
            if self.at and self.norm_emb and token_weight is not None:
                weights = tf.constant(np.load(token_weight)["embeddings"], dtype=tf.float32, name="weight",
                                      shape=[token_size - 1, 1])
                table = self.emb_normalize(table, weights)
            self.table = tf.concat([tf.zeros([1, token_dim], dtype=tf.float32), table], axis=0)
            if self.word_project:
                self.dense = tf.layers.Dense(units=token_dim, use_bias=True, _reuse=tf.AUTO_REUSE, name="word_project")

    @staticmethod
    def emb_normalize(emb, weights):
        mean = tf.reduce_sum(weights * emb, axis=0, keepdims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.0), axis=0, keepdims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev

    def __call__(self, tokens):
        with tf.variable_scope(self.scope):
            token_emb = tf.nn.embedding_lookup(self.table, tokens)
            if self.word_project:
                token_emb = self.dense(token_emb)
            return token_emb


def self_attn(inputs, return_alphas=False, project=True, reuse=None, name="self_attention"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        hidden_size = inputs.shape[-1].value
        if project:
            x = tf.layers.dense(inputs, units=hidden_size, use_bias=False, activation=tf.nn.tanh)
        else:
            x = inputs
        weight = tf.get_variable(name="weight", shape=[hidden_size, 1], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.01, seed=1227))
        x = tf.tensordot(x, weight, axes=1)
        alphas = tf.nn.softmax(x, axis=-1)
        output = tf.matmul(tf.transpose(inputs, perm=[0, 2, 1]), alphas)
        output = tf.squeeze(output, axis=-1)
        if return_alphas:
            return output, alphas
        else:
            return output


def focal_loss(logits, labels, weights=None, alpha=0.25, gamma=2):
    logits = tf.nn.softmax(logits, axis=1)  # logits = tf.nn.sigmoid(logits)
    labels = tf.cast(labels, dtype=tf.float32)
    if labels.get_shape().ndims < logits.get_shape().ndims:
        labels = tf.one_hot(labels, depth=logits.shape[-1].value, axis=-1)
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    pos_logits_prob = tf.where(labels > zeros, labels - logits, zeros)
    neg_logits_prob = tf.where(labels > zeros, zeros, logits)
    cross_entropy = - alpha * (pos_logits_prob ** gamma) * tf.log(tf.clip_by_value(logits, 1e-8, 1.0)) \
                    - (1 - alpha) * (neg_logits_prob ** gamma) * tf.log(tf.clip_by_value(1.0 - logits, 1e-8, 1.0))
    if weights is not None:
        if weights.get_shape().ndims < logits.get_shape().ndims:
            weights = tf.expand_dims(weights, axis=-1)
        cross_entropy = cross_entropy * weights
    return cross_entropy


class Base:
    def __init__(self, config):
        self.cfg = config
        tf.set_random_seed(self.cfg.random_seed)
        # create folders and logger
        if not os.path.exists(self.cfg.checkpoint_path):
            os.makedirs(self.cfg.checkpoint_path)
        self.logger = get_logger(os.path.join(self.cfg.checkpoint_path, "log.txt"))

    def _initialize_session(self):
        if not self.cfg.use_gpu:
            self.sess = tf.Session()
        else:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=self.cfg.max_to_keep)
        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.cfg.checkpoint_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.cfg.checkpoint_path + self.cfg.model_name, global_step=epoch)

    def close_session(self):
        self.sess.close()

    @staticmethod
    def viterbi_decode(logits, trans_params, seq_len):
        viterbi_sequences = []
        for logit, lens in zip(logits, seq_len):
            logit = logit[:lens]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]
        return viterbi_sequences

    @staticmethod
    def emb_normalize(emb, weights):
        mean = tf.reduce_sum(weights * emb, axis=0, keepdims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.0), axis=0, keepdims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev

    @staticmethod
    def add_perturbation(emb, loss, epsilon=5.0):
        """Adds gradient to embedding and recomputes classification loss."""
        grad, = tf.gradients(loss, emb, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        alpha = tf.reduce_max(tf.abs(grad), axis=(1, 2), keepdims=True) + 1e-12  # l2 scale
        l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(grad / alpha, 2), axis=(1, 2), keepdims=True) + 1e-6)
        norm_grad = grad / l2_norm
        perturb = epsilon * norm_grad
        return emb + perturb

    @staticmethod
    def count_params(scope=None):
        if scope is None:
            return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        else:
            return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope)])

    @staticmethod
    def load_dataset(filename):
        with codecs.open(filename, mode='r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset

    def _add_summary(self, summary_path):
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(summary_path + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(summary_path + "test")

    @staticmethod
    def _arrange_batches(src_dataset, tgt_dataset, mix_rate, train_ratio):
        r_tgt = math.sqrt(tgt_dataset.org_train_dataset_size)
        r_src = math.sqrt(src_dataset.org_train_dataset_size) * mix_rate
        num_src = int(src_dataset.get_num_batches() * r_src / r_tgt)
        if tgt_dataset.get_dataset_size() <= 500:  # small tricks
            num_tgt = math.ceil(tgt_dataset.get_num_batches() * 0.2)
            batches = ["src"] * num_src + ["tgt"] * num_tgt
        elif train_ratio == "1.0":
            num_tgt = math.ceil(tgt_dataset.get_num_batches() * 0.7)
            batches = ["src"] * num_src + ["tgt"] * num_tgt
        else:
            num_tgt = math.ceil(tgt_dataset.get_num_batches() * 0.5)
            batches = ["src"] * num_src + ["tgt"] * num_tgt
        random.shuffle(batches)
        batches += ["tgt"] * (tgt_dataset.get_num_batches() - num_tgt)
        return batches

    def evaluate_f1(self, dataset, rev_word_dict, rev_label_dict, name):
        save_path = os.path.join(self.cfg.checkpoint_path, name + "_result.txt")
        if os.path.exists(save_path):
            os.remove(save_path)
        predictions, groundtruth, words_list = list(), list(), list()
        for b_labels, b_predicts, b_words, b_seq_len in dataset:
            for labels, predicts, words, seq_len in zip(b_labels, b_predicts, b_words, b_seq_len):
                predictions.append([rev_label_dict[x] for x in predicts[:seq_len]])
                groundtruth.append([rev_label_dict[x] for x in labels[:seq_len]])
                words_list.append([rev_word_dict[x] for x in words[:seq_len]])
        conll_eval = CoNLLeval()
        score = conll_eval.conlleval(predictions, groundtruth, words_list, save_path)
        self.logger.info("{} dataset -- acc: {:04.2f}, pre: {:04.2f}, rec: {:04.2f}, FB1: {:04.2f}"
                         .format(name, score["accuracy"], score["precision"], score["recall"], score["FB1"]))
        return score

    def evaluate(self, dataset, name, predict_op, rev_word_dict, rev_label_dict):
        all_data = list()
        for data in dataset:
            predicts = predict_op(data)
            all_data.append((data["labels"], predicts, data["words"], data["seq_len"]))
        return self.evaluate_f1(all_data, rev_word_dict, rev_label_dict, name)


def tsne_reduce(vecs, n_components=2):
    # perplexity is between [5, 50], for large data, using large value and vice versa
    tsne = TSNE(perplexity=30, n_components=n_components, init="pca", n_iter=1000, method="exact")
    vecs_reduced = tsne.fit_transform(vecs)
    return vecs_reduced
