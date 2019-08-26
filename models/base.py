import tensorflow as tf
import numpy as np
from models.shares import Base, BiRNN, CharTDNNHW, CRF, Embedding
from utils.logger import Progbar


class BaseModel(Base):
    """Basement model: Char CNN + LSTM encoder + Chain CRF decoder module"""
    def __init__(self, config):
        super(BaseModel, self).__init__(config)
        self._init_configs()
        with tf.Graph().as_default():
            self._add_placeholders()
            self._build_model()
            self.logger.info("total params: {}".format(self.count_params()))
            self._initialize_session()

    def _init_configs(self):
        vocab = self.load_dataset(self.cfg.vocab)
        self.word_dict, self.char_dict, self.label_dict = vocab["word_dict"], vocab["char_dict"], vocab["label_dict"]
        self.word_size, self.char_size, self.label_size = len(self.word_dict), len(self.char_dict), len(self.label_dict)
        self.rev_word_dict = dict([(idx, word) for word, idx in self.word_dict.items()])
        self.rev_char_dict = dict([(idx, char) for char, idx in self.char_dict.items()])
        self.rev_label_dict = dict([(idx, tag) for tag, idx in self.label_dict.items()])

    def _get_feed_dict(self, data, is_train=False, lr=None):
        feed_dict = {self.words: data["words"], self.seq_len: data["seq_len"], self.chars: data["chars"],
                     self.char_seq_len: data["char_seq_len"]}
        if "labels" in data:
            feed_dict[self.labels] = data["labels"]
        feed_dict[self.is_train] = is_train
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _add_placeholders(self):
        self.words = tf.placeholder(tf.int32, shape=[None, None], name="words")
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
        self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name="chars")
        self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

    def _build_model(self):
        with tf.variable_scope("embeddings_op"):
            # word table
            word_table = Embedding(self.word_size, self.cfg.word_dim, self.cfg.wordvec, self.cfg.word_weight,
                                   self.cfg.tune_emb, self.cfg.at, self.cfg.norm_emb, self.cfg.word_project,
                                   scope="word_table")
            word_emb = word_table(self.words)
            # char table
            char_table = Embedding(self.char_size, self.cfg.char_dim, None, self.cfg.char_weight, True, self.cfg.at,
                                   self.cfg.norm_emb, False, scope="char_table")
            char_emb = char_table(self.chars)

        with tf.variable_scope("computation_graph"):
            # create module
            emb_dropout = tf.layers.Dropout(rate=self.cfg.emb_drop_rate)
            char_tdnn_hw = CharTDNNHW(self.cfg.char_kernels, self.cfg.char_kernel_features, self.cfg.char_dim,
                                      self.cfg.highway_layers, padding="VALID", activation=tf.tanh, use_bias=True,
                                      hw_activation=tf.tanh, reuse=tf.AUTO_REUSE, scope="char_tdnn_hw")
            bi_rnn = BiRNN(self.cfg.num_units, drop_rate=self.cfg.rnn_drop_rate, concat=self.cfg.concat_rnn,
                           activation=tf.tanh, reuse=tf.AUTO_REUSE, scope="bi_rnn")
            dense = tf.layers.Dense(units=self.label_size, use_bias=True, _reuse=tf.AUTO_REUSE, name="project")
            crf_layer = CRF(self.label_size, reuse=tf.AUTO_REUSE, scope="crf")

            # compute outputs
            def compute_logits(w_emb, c_emb):
                char_cnn = char_tdnn_hw(c_emb)
                emb = emb_dropout(tf.concat([w_emb, char_cnn], axis=-1), training=self.is_train)
                rnn_outputs = bi_rnn(emb, self.seq_len, training=self.is_train)
                logits = dense(rnn_outputs)
                transition, mean_loss = crf_layer(logits, self.labels, self.seq_len)
                return logits, transition, mean_loss

            self.logits, self.transition, self.loss = compute_logits(word_emb, char_emb)
            if self.cfg.at:
                perturb_word_emb = self.add_perturbation(word_emb, self.loss, epsilon=self.cfg.epsilon)
                perturb_char_emb = self.add_perturbation(char_emb, self.loss, epsilon=self.cfg.epsilon)
                *_, adv_loss = compute_logits(perturb_word_emb, perturb_char_emb)
                self.loss += adv_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            self.train_op = optimizer.minimize(self.loss)

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        logits, transition, seq_len = self.sess.run([self.logits, self.transition, self.seq_len], feed_dict=feed_dict)
        return self.viterbi_decode(logits, transition, seq_len)

    def train(self, dataset):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch, init_lr, lr, cur_step = -np.inf, 0, self.cfg.lr, self.cfg.lr, 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.logger.info('Epoch {}/{}:'.format(epoch, self.cfg.epochs))
            prog = Progbar(target=dataset.get_num_batches())
            for i, data in enumerate(dataset.get_data_batches()):
                cur_step += 1
                feed_dict = self._get_feed_dict(data, is_train=True, lr=lr)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            # learning rate decay
            if self.cfg.use_lr_decay:
                if self.cfg.decay_step:
                    lr = max(init_lr / (1.0 + self.cfg.lr_decay * epoch / self.cfg.decay_step), self.cfg.minimal_lr)
            # evaluate
            if not self.cfg.dev_for_train:
                self.evaluate(dataset.get_data_batches("dev"), "dev", self._predict_op, self.rev_word_dict,
                              self.rev_label_dict)
            score = self.evaluate(dataset.get_data_batches("test"), "test", self._predict_op, self.rev_word_dict,
                                  self.rev_label_dict)
            if score["FB1"] > best_f1:
                best_f1, no_imprv_epoch = score["FB1"], 0
                self.save_session(epoch)
                self.logger.info(' -- new BEST score on test dataset: {:04.2f}'.format(best_f1))
            else:
                no_imprv_epoch += 1
                if self.cfg.no_imprv_tolerance is not None and no_imprv_epoch >= self.cfg.no_imprv_tolerance:
                    self.logger.info('early stop at {}th epoch without improvement'.format(epoch))
                    self.logger.info('best score on test set: {}'.format(best_f1))
                    break

    def evaluate_data(self, dataset, name):
        self.evaluate(dataset, name, self._predict_op, self.rev_word_dict, self.rev_label_dict)
