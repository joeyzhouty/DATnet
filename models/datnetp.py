import tensorflow as tf
import numpy as np
import os
from models.shares import Base, flip_gradient, CharTDNNHW, BiRNN, CRF, Embedding, self_attn, focal_loss, tsne_reduce
from utils.plot_graph import plot_scatter_graph
from utils.logger import Progbar


class DATNetPModel(Base):
    """Part Transfer and DATNet-P modules"""
    def __init__(self, config):
        super(DATNetPModel, self).__init__(config)
        self._init_configs()
        with tf.Graph().as_default():
            self._add_placeholders()
            self._build_model()
            self.logger.info("total params: {}".format(self.count_params()))
            self._initialize_session()

    def _init_configs(self):
        s_vocab = self.load_dataset(self.cfg.src_vocab)
        t_vocab = self.load_dataset(self.cfg.tgt_vocab)
        self.sw_dict, self.sc_dict, self.sl_dict = s_vocab["word_dict"], s_vocab["char_dict"], s_vocab["label_dict"]
        self.tw_dict, self.tc_dict, self.tl_dict = t_vocab["word_dict"], t_vocab["char_dict"], t_vocab["label_dict"]
        del s_vocab, t_vocab
        self.sw_size, self.sc_size, self.sl_size = len(self.sw_dict), len(self.sc_dict), len(self.sl_dict)
        self.tw_size, self.tc_size, self.tl_size = len(self.tw_dict), len(self.tc_dict), len(self.tl_dict)
        self.rev_sw_dict = dict([(idx, word) for word, idx in self.sw_dict.items()])
        self.rev_sc_dict = dict([(idx, char) for char, idx in self.sc_dict.items()])
        self.rev_sl_dict = dict([(idx, label) for label, idx in self.sl_dict.items()])
        self.rev_tw_dict = dict([(idx, word) for word, idx in self.tw_dict.items()])
        self.rev_tc_dict = dict([(idx, char) for char, idx in self.tc_dict.items()])
        self.rev_tl_dict = dict([(idx, label) for label, idx in self.tl_dict.items()])

    def _get_feed_dict(self, src_data, tgt_data, domain_labels, is_train=False, src_lr=None, tgt_lr=None):
        feed_dict = {self.is_train: is_train}
        if src_lr is not None:
            feed_dict[self.src_lr] = src_lr
        if tgt_lr is not None:
            feed_dict[self.tgt_lr] = tgt_lr
        if src_data is not None:
            feed_dict[self.src_words] = src_data["words"]
            feed_dict[self.src_seq_len] = src_data["seq_len"]
            feed_dict[self.src_chars] = src_data["chars"]
            feed_dict[self.src_char_seq_len] = src_data["char_seq_len"]
            if "labels" in src_data:
                feed_dict[self.src_labels] = src_data["labels"]
        if tgt_data is not None:
            feed_dict[self.tgt_words] = tgt_data["words"]
            feed_dict[self.tgt_seq_len] = tgt_data["seq_len"]
            feed_dict[self.tgt_chars] = tgt_data["chars"]
            feed_dict[self.tgt_char_seq_len] = tgt_data["char_seq_len"]
            if "labels" in tgt_data:
                feed_dict[self.tgt_labels] = tgt_data["labels"]
        if domain_labels is not None:
            feed_dict[self.domain_labels] = domain_labels
        return feed_dict

    def _add_placeholders(self):
        # source placeholders
        self.src_words = tf.placeholder(tf.int32, shape=[None, None], name="source_words")
        self.src_seq_len = tf.placeholder(tf.int32, shape=[None], name="source_seq_len")
        self.src_chars = tf.placeholder(tf.int32, shape=[None, None, None], name="source_chars")
        self.src_char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="source_char_seq_len")
        self.src_labels = tf.placeholder(tf.int32, shape=[None, None], name="source_labels")
        # target placeholders
        self.tgt_words = tf.placeholder(tf.int32, shape=[None, None], name="target_words")
        self.tgt_seq_len = tf.placeholder(tf.int32, shape=[None], name="target_seq_len")
        self.tgt_chars = tf.placeholder(tf.int32, shape=[None, None, None], name="target_chars")
        self.tgt_char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="target_char_seq_len")
        self.tgt_labels = tf.placeholder(tf.int32, shape=[None, None], name="target_labels")
        # domain labels
        self.domain_labels = tf.placeholder(tf.int32, shape=[None, 2], name="domain_labels")
        # hyper-parameters
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.src_lr = tf.placeholder(tf.float32, name="source_learning_rate")
        self.tgt_lr = tf.placeholder(tf.float32, name="target_learning_rate")

    def _build_model(self):
        with tf.variable_scope("embeddings_op"):
            # word table
            if self.cfg.share_word:
                word_table = Embedding(self.sw_size, self.cfg.src_word_dim, self.cfg.src_wordvec,
                                       self.cfg.src_word_weight, self.cfg.tune_emb, self.cfg.at, self.cfg.norm_emb,
                                       self.cfg.word_project, scope="word_table")
                src_word_emb = word_table(self.src_words)
                tgt_word_emb = word_table(self.tgt_words)
            else:
                src_word_table = Embedding(self.sw_size, self.cfg.src_word_dim, self.cfg.src_wordvec,
                                           self.cfg.src_word_weight, self.cfg.tune_emb, self.cfg.at, self.cfg.norm_emb,
                                           self.cfg.word_project, scope="source_word_table")
                src_word_emb = src_word_table(self.src_words)
                tgt_word_table = Embedding(self.tw_size, self.cfg.tgt_word_dim, self.cfg.tgt_wordvec,
                                           self.cfg.tgt_word_weight, self.cfg.tune_emb, self.cfg.at, self.cfg.norm_emb,
                                           self.cfg.word_project, scope="target_word_table")
                tgt_word_emb = tgt_word_table(self.tgt_words)
            # char table (default char is shared)
            char_table = Embedding(self.sc_size, self.cfg.char_dim, None, self.cfg.char_weight, True, self.cfg.at,
                                   self.cfg.norm_emb, False, scope="char_table")
            src_char_emb = char_table(self.src_chars)
            tgt_char_emb = char_table(self.tgt_chars)

        with tf.variable_scope("computation_graph"):
            # build module
            emb_dropout = tf.layers.Dropout(rate=self.cfg.emb_drop_rate)
            char_tdnn_hw = CharTDNNHW(self.cfg.char_kernels, self.cfg.char_kernel_features, self.cfg.char_dim,
                                      self.cfg.highway_layers, padding="VALID", activation=tf.nn.tanh, use_bias=True,
                                      hw_activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope="char_tdnn_hw")
            src_bi_rnn = BiRNN(self.cfg.src_num_units, drop_rate=self.cfg.rnn_drop_rate, concat=self.cfg.concat_rnn,
                               activation=tf.tanh, reuse=tf.AUTO_REUSE, scope="src_bi_rnn")
            tgt_bi_rnn = BiRNN(self.cfg.tgt_num_units, drop_rate=self.cfg.rnn_drop_rate, concat=self.cfg.concat_rnn,
                               activation=tf.tanh, reuse=tf.AUTO_REUSE, scope="tgt_bi_rnn")
            share_bi_rnn = BiRNN(self.cfg.share_num_units, drop_rate=self.cfg.rnn_drop_rate, concat=self.cfg.concat_rnn,
                                 activation=tf.tanh, reuse=tf.AUTO_REUSE, scope="share_bi_rnn")
            # create dense layer
            if self.cfg.share_dense:
                src_dense = tf.layers.Dense(self.sl_size, use_bias=True, _reuse=tf.AUTO_REUSE, name="project")
                tgt_dense = tf.layers.Dense(self.tl_size, use_bias=True, _reuse=tf.AUTO_REUSE, name="project")
            else:
                src_dense = tf.layers.Dense(self.sl_size, use_bias=True, _reuse=tf.AUTO_REUSE, name="src_project")
                tgt_dense = tf.layers.Dense(self.tl_size, use_bias=True, _reuse=tf.AUTO_REUSE, name="tgt_project")
            # create CRF layer
            if self.cfg.share_label:
                src_crf_layer = CRF(self.sl_size, reuse=tf.AUTO_REUSE, scope="crf_layer")
                tgt_crf_layer = CRF(self.sl_size, reuse=tf.AUTO_REUSE, scope="crf_layer")
            else:
                src_crf_layer = CRF(self.sl_size, reuse=tf.AUTO_REUSE, scope="src_crf_layer")
                tgt_crf_layer = CRF(self.tl_size, reuse=tf.AUTO_REUSE, scope="tgt_crf_layer")

            # compute outputs
            def discriminator(feature):
                feat = flip_gradient(feature, lw=self.cfg.grad_rev_rate)
                outputs = self_attn(feat, reuse=tf.AUTO_REUSE, name="self_attention")
                logits = tf.layers.dense(outputs, units=2, use_bias=True, name="disc_project", reuse=tf.AUTO_REUSE)
                if self.cfg.discriminator == 2:  # GRAD
                    loss = focal_loss(logits, self.domain_labels, alpha=self.cfg.alpha, gamma=self.cfg.gamma)
                    loss = tf.reduce_mean(loss)
                else:  # normal discriminator
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.domain_labels)
                    loss = tf.reduce_mean(loss)
                return loss

            def compute_src_logits(word_emb, char_emb):
                char_cnn = char_tdnn_hw(char_emb)
                emb = emb_dropout(tf.concat([word_emb, char_cnn], axis=-1), training=self.is_train)
                rnn_outs = src_bi_rnn(emb, self.src_seq_len, training=self.is_train)
                share_rnn_outs = share_bi_rnn(emb, self.src_seq_len, training=self.is_train)
                rnn_outs = tf.concat([rnn_outs, share_rnn_outs], axis=-1)
                logits = src_dense(rnn_outs)
                transition, mean_loss = src_crf_layer(logits, self.src_labels, self.src_seq_len)
                return share_rnn_outs, logits, transition, mean_loss

            def compute_tgt_logits(word_emb, char_emb):
                char_cnn = char_tdnn_hw(char_emb)
                emb = emb_dropout(tf.concat([word_emb, char_cnn], axis=-1), training=self.is_train)
                rnn_outs = tgt_bi_rnn(emb, self.tgt_seq_len, training=self.is_train)
                share_rnn_outs = share_bi_rnn(emb, self.tgt_seq_len, training=self.is_train)
                rnn_outs = tf.concat([rnn_outs, share_rnn_outs], axis=-1)
                logits = tgt_dense(rnn_outs)
                transition, mean_loss = tgt_crf_layer(logits, self.tgt_labels, self.tgt_seq_len)
                return share_rnn_outs, logits, transition, mean_loss

            # train source
            self.src_share_rnn_outs, self.src_logits, self.src_transition, self.src_loss = compute_src_logits(
                src_word_emb, src_char_emb)
            if self.cfg.at:  # adversarial training
                perturb_src_word_emb = self.add_perturbation(src_word_emb, self.src_loss, epsilon=self.cfg.epsilon)
                perturb_src_char_emb = self.add_perturbation(src_char_emb, self.src_loss, epsilon=self.cfg.epsilon)
                *_, adv_src_loss = compute_src_logits(perturb_src_word_emb, perturb_src_char_emb)
                self.src_loss = self.src_loss + adv_src_loss
            if self.cfg.discriminator != 0:  # if 0 means no discriminator is applied
                src_dis_loss = discriminator(self.src_share_rnn_outs)
                self.src_loss += src_dis_loss

            # train target
            self.tgt_share_rnn_outs, self.tgt_logits, self.tgt_transition, self.tgt_loss = compute_tgt_logits(
                tgt_word_emb, tgt_char_emb)
            if self.cfg.at:  # adversarial training
                perturb_tgt_word_emb = self.add_perturbation(tgt_word_emb, self.tgt_loss, epsilon=self.cfg.epsilon)
                perturb_tgt_char_emb = self.add_perturbation(tgt_char_emb, self.tgt_loss, epsilon=self.cfg.epsilon)
                *_, adv_tgt_loss = compute_tgt_logits(perturb_tgt_word_emb, perturb_tgt_char_emb)
                self.tgt_loss = self.tgt_loss + adv_tgt_loss
            if self.cfg.discriminator != 0:  # if 0 means no discriminator is applied
                tgt_dis_loss = discriminator(self.tgt_share_rnn_outs)
                self.tgt_loss = self.tgt_loss + tgt_dis_loss

        src_optimizer = tf.train.AdamOptimizer(learning_rate=self.src_lr)
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            grads, vs = zip(*src_optimizer.compute_gradients(self.src_loss))
            grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
            self.src_train_op = src_optimizer.apply_gradients(zip(grads, vs))
        else:
            self.src_train_op = src_optimizer.minimize(self.src_loss)

        tgt_optimizer = tf.train.AdamOptimizer(learning_rate=self.tgt_lr)
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            grads, vs = zip(*tgt_optimizer.compute_gradients(self.tgt_loss))
            grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
            self.tgt_train_op = tgt_optimizer.apply_gradients(zip(grads, vs))
        else:
            self.tgt_train_op = tgt_optimizer.minimize(self.tgt_loss)

    def _src_predict_op(self, data):
        feed_dict = self._get_feed_dict(src_data=data, tgt_data=None, domain_labels=None)
        logits, transition, seq_len = self.sess.run([self.src_logits, self.src_transition, self.src_seq_len],
                                                    feed_dict=feed_dict)
        return self.viterbi_decode(logits, transition, seq_len)

    def _tgt_predict_op(self, data):
        feed_dict = self._get_feed_dict(src_data=None, tgt_data=data, domain_labels=None)
        logits, transition, seq_len = self.sess.run([self.tgt_logits, self.tgt_transition, self.tgt_seq_len],
                                                    feed_dict=feed_dict)
        return self.viterbi_decode(logits, transition, seq_len)

    def train(self, src_dataset, tgt_dataset):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch, src_lr, tgt_lr, cur_step = -np.inf, 0, self.cfg.lr, self.cfg.lr, 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, self.cfg.epochs))
            batches = self._arrange_batches(src_dataset, tgt_dataset, self.cfg.mix_rate, self.cfg.train_ratio)
            prog = Progbar(target=len(batches))
            prog.update(0, [("Global Step", int(cur_step)), ("Source Train Loss", 0.0), ("Target Train Loss", 0.0)])
            for i, batch_name in enumerate(batches):
                cur_step += 1
                if batch_name == "src":
                    data = src_dataset.get_next_train_batch()
                    domain_labels = [[1, 0]] * data["batch_size"]
                    feed_dict = self._get_feed_dict(src_data=data, tgt_data=None, domain_labels=domain_labels,
                                                    is_train=True, src_lr=src_lr)
                    _, src_cost = self.sess.run([self.src_train_op, self.src_loss], feed_dict=feed_dict)
                    prog.update(i + 1, [("Global Step", int(cur_step)), ("Source Train Loss", src_cost)])
                else:  # "tgt"
                    data = tgt_dataset.get_next_train_batch()
                    domain_labels = [[0, 1]] * data["batch_size"]
                    feed_dict = self._get_feed_dict(src_data=None, tgt_data=data, domain_labels=domain_labels,
                                                    is_train=True, tgt_lr=tgt_lr)
                    _, tgt_cost = self.sess.run([self.tgt_train_op, self.tgt_loss], feed_dict=feed_dict)
                    prog.update(i + 1, [("Global Step", int(cur_step)), ("Target Train Loss", tgt_cost)])
            if self.cfg.use_lr_decay:  # source learning rate decay
                src_lr = max(self.cfg.lr / (1.0 + self.cfg.lr_decay * epoch), self.cfg.minimal_lr)
                if epoch % self.cfg.decay_step == 0:
                    tgt_lr = max(self.cfg.lr / (1.0 + self.cfg.lr_decay * epoch / self.cfg.decay_step),
                                 self.cfg.minimal_lr)
            if not self.cfg.dev_for_train:
                self.evaluate(tgt_dataset.get_data_batches("dev"), "target_dev", self._tgt_predict_op,
                              rev_word_dict=self.rev_tw_dict, rev_label_dict=self.rev_tl_dict)
            score = self.evaluate(tgt_dataset.get_data_batches("test"), "target_test", self._tgt_predict_op,
                                  rev_word_dict=self.rev_tw_dict, rev_label_dict=self.rev_tl_dict)
            if score["FB1"] > best_f1:
                best_f1, no_imprv_epoch = score["FB1"], 0
                self.save_session(epoch)
                self.logger.info(' -- new BEST score on target test dataset: {:04.2f}'.format(best_f1))
            else:
                no_imprv_epoch += 1
                if self.cfg.no_imprv_tolerance is not None and no_imprv_epoch >= self.cfg.no_imprv_tolerance:
                    self.logger.info('early stop at {}th epoch without improvement'.format(epoch))
                    self.logger.info('best score on target test set: {}'.format(best_f1))
                    break

    def evaluate_data(self, dataset, name, resource="target"):
        if resource == "target":
            self.evaluate(dataset, name, self._tgt_predict_op, self.rev_tw_dict, self.rev_tl_dict)
        else:
            self.evaluate(dataset, name, self._src_predict_op, self.rev_sw_dict, self.rev_sl_dict)

    # Experimental code for Shared RNN output features extraction
    def get_shared_rnn_features(self, src_dataset, tgt_dataset):
        src_share, tgt_share = [], []
        # process src dataset
        src_batches = src_dataset.get_data_batches("test")
        for data in src_batches:
            feed_dict = self._get_feed_dict(src_data=data, tgt_data=None, domain_labels=None)
            share_rnn_outs, mask = self.sess.run(
                [tf.reshape(self.src_share_rnn_outs, shape=[-1, 2 * self.cfg.share_num_units]),
                 tf.reshape(tf.sequence_mask(self.src_seq_len, maxlen=tf.reduce_max(self.src_seq_len), dtype=tf.int32),
                            shape=[-1])], feed_dict=feed_dict)
            for i in range(share_rnn_outs.shape[0]):
                if mask[i] != 0:
                    src_share.append(share_rnn_outs[i])
        np.savez_compressed("src_share_features.npz", embeddings=np.asarray(src_share))
        # process tgt dataset
        tgt_batches = tgt_dataset.get_data_batches("test")
        for data in tgt_batches:
            feed_dict = self._get_feed_dict(src_data=None, tgt_data=data, domain_labels=None)
            share_rnn_outs, mask = self.sess.run(
                [tf.reshape(self.tgt_share_rnn_outs, shape=[-1, 2 * self.cfg.share_num_units]),
                 tf.reshape(tf.sequence_mask(self.tgt_seq_len, maxlen=tf.reduce_max(self.tgt_seq_len), dtype=tf.int32),
                            shape=[-1])], feed_dict=feed_dict)
            for i in range(share_rnn_outs.shape[0]):
                if mask[i] != 0:
                    tgt_share.append(share_rnn_outs[i])
        np.savez_compressed("tgt_share_features.npz", embeddings=np.asarray(tgt_share))

    @staticmethod
    def tsne_dim_reduction(num_features=10000, save_fig=True, save_format="pdf", save_name="GRAD"):
        src_share_feature = np.load("src_share_features.npz")["embeddings"]
        tgt_share_feature = np.load("tgt_share_features.npz")["embeddings"]
        if os.path.exists("src_share_features_reduced.npz") and os.path.exists("tgt_share_features_reduced.npz"):
            src_share_feature_reduced = np.load("src_share_features_reduced.npz")["embeddings"]
            tgt_share_feature_reduced = np.load("tgt_share_features_reduced.npz")["embeddings"]
        else:
            src_share_feature_reduced = tsne_reduce(src_share_feature[0:num_features])
            tgt_share_feature_reduced = tsne_reduce(tgt_share_feature[0:num_features])
            np.savez_compressed("src_share_features_reduced.npz", embeddings=src_share_feature_reduced)
            np.savez_compressed("tgt_share_features_reduced.npz", embeddings=tgt_share_feature_reduced)
        plot_scatter_graph(src_share_feature_reduced, tgt_share_feature_reduced, save_fig=save_fig,
                           save_format=save_format, save_name=save_name)
