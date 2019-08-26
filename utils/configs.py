import os
from utils.data_utils import EMB_PATH, EMB_DICT, RAW_DICT, SRC_TGT


class Configurations:
    def __init__(self, parser):
        # set common arguments
        self.use_gpu, self.gpu_idx, self.random_seed = parser.use_gpu, parser.gpu_idx, parser.random_seed
        self.train, self.restore_model, self.iobes = parser.train, parser.restore_model, parser.iobes
        self.at, self.epsilon, self.norm_emb = parser.at, parser.epsilon, parser.norm_emb
        self.train_ratio, self.threshold = parser.train_ratio, parser.threshold
        self.dev_for_train = parser.dev_for_train
        self.concat_rnn, self.word_project = parser.concat_rnn, parser.word_project
        self.word_lowercase, self.char_lowercase = parser.word_lowercase, parser.char_lowercase
        self.char_dim, self.tune_emb, self.highway_layers = parser.char_dim, parser.tune_emb, parser.highway_layers
        self.char_kernels, self.char_kernel_features = parser.char_kernels, parser.char_kernel_features
        self.lr, self.use_lr_decay, self.lr_decay = parser.lr, parser.use_lr_decay, parser.lr_decay
        self.decay_step, self.minimal_lr, self.optimizer = parser.decay_step, parser.minimal_lr, parser.optimizer
        self.grad_clip, self.epochs, self.batch_size = parser.grad_clip, parser.epochs, parser.batch_size
        self.emb_drop_rate, self.rnn_drop_rate = parser.emb_drop_rate, parser.rnn_drop_rate
        self.max_to_keep, self.no_imprv_tolerance = parser.max_to_keep, parser.no_imprv_tolerance
        self.restore_best = parser.restore_best
        if self.restore_best:
            self.train = False
            self.restore_model = False
        if parser.mode == 0:  # base model settings
            self._base_configs(parser)
        elif parser.mode == 1:  # partially transfer settings (DATNet-P)
            self._transfer_configs(parser)
        elif parser.mode == 2:  # fully transfer settings (DATNet-F)
            self._transfer_v2_configs(parser)
        else:
            raise ValueError("Error mode!!! Only support mode [0, 1, 2]")

    def _base_configs(self, parser):
        self.word_dim, self.language = parser.word_dim, parser.language
        self.num_units = parser.num_units
        self.model_name = "adv_model_{}".format(self.language) if self.at else "base_model_{}".format(self.language)
        r_path = "datasets/raw/{}/".format(RAW_DICT[self.language])
        self.train_file, self.dev_file, self.test_file = r_path + "train.txt", r_path + "valid.txt", r_path + "test.txt"
        if self.restore_best:
            self.save_path = "datasets/best_data/{}/".format(self.model_name)
        else:
            self.save_path = "datasets/data/{}/".format(self.model_name)
        self.train_set, self.dev_set = self.save_path + "train.json", self.save_path + "dev.json"
        self.test_set, self.vocab = self.save_path + "test.json", self.save_path + "vocab.json"
        self.word_weight, self.char_weight = self.save_path + "word_weight.npz", self.save_path + "char_weight.npz"
        self.wordvec_path = os.path.join(EMB_PATH, EMB_DICT[self.language].format(self.word_dim))
        self.wordvec = self.save_path + "wordvec.npz"
        if self.restore_best:
            self.checkpoint_path = "best_ckpt/{}_{}/".format(self.model_name, self.train_ratio)
        else:
            self.checkpoint_path = "ckpt/{}_{}/".format(self.model_name, self.train_ratio)
        self.summary_path = self.checkpoint_path + "summary/"

    def _transfer_configs(self, parser):
        self.src_word_dim, self.tgt_word_dim = parser.src_word_dim, parser.tgt_word_dim
        self.src_language, self.tgt_language = parser.src_language, parser.tgt_language
        self.share_word, self.share_dense, self.share_label = parser.share_word, parser.share_dense, parser.share_label
        self.src_num_units, self.tgt_num_units = parser.src_num_units, parser.tgt_num_units
        self.share_num_units, self.mix_rate = parser.share_num_units, parser.mix_rate
        self.discriminator, self.alpha, self.gamma = parser.discriminator, parser.alpha, parser.gamma
        self.grad_rev_rate = parser.grad_rev_rate
        if self.at:
            self.model_name = "adv_trans_model_{}".format(SRC_TGT[self.tgt_language])
        else:
            self.model_name = "trans_model_{}".format(SRC_TGT[self.tgt_language])
        if self.restore_best:
            self.checkpoint_path = "best_ckpt/{}_{}/".format(self.model_name, self.train_ratio)
        else:
            self.checkpoint_path = "ckpt/{}_{}/".format(self.model_name, self.train_ratio)
        self.summary_path = self.checkpoint_path + "summary/"
        self._transfer_data_config()

    def _transfer_v2_configs(self, parser):
        self.src_word_dim, self.tgt_word_dim = parser.src_word_dim, parser.tgt_word_dim
        self.src_language, self.tgt_language = parser.src_language, parser.tgt_language
        self.share_word, self.share_dense, self.share_label = parser.share_word, parser.share_dense, parser.share_label
        self.num_units, self.mix_rate = parser.num_units, parser.mix_rate
        self.discriminator, self.alpha, self.gamma = parser.discriminator, parser.alpha, parser.gamma
        self.grad_rev_rate = parser.grad_rev_rate
        if self.at:
            self.model_name = "adv_trans_model_v2_{}".format(SRC_TGT[self.tgt_language])
        else:
            self.model_name = "trans_model_v2_{}".format(SRC_TGT[self.tgt_language])
        if self.restore_best:
            self.checkpoint_path = "best_ckpt/{}_{}/".format(self.model_name, self.train_ratio)
        else:
            self.checkpoint_path = "ckpt/{}_{}/".format(self.model_name, self.train_ratio)
        self.summary_path = self.checkpoint_path + "summary/"
        self._transfer_data_config()

    def _transfer_data_config(self):
        sr_path = "datasets/raw/{}/".format(RAW_DICT[self.src_language])
        self.src_train_file, self.src_dev_file = sr_path + "train.txt", sr_path + "valid.txt"
        self.src_test_file = sr_path + "test.txt"
        tr_path = "datasets/raw/{}/".format(RAW_DICT[self.tgt_language])
        self.tgt_train_file, self.tgt_dev_file = tr_path + "train.txt", tr_path + "valid.txt"
        self.tgt_test_file = tr_path + "test.txt"
        if self.restore_best:
            self.save_path = "datasets/best_data/{}/".format(self.model_name)
        else:
            self.save_path = "datasets/data/{}/".format(self.model_name)
        self.src_train_set, self.src_dev_set = self.save_path + "src_train.json", self.save_path + "src_dev.json"
        self.src_test_set, self.src_vocab = self.save_path + "src_test.json", self.save_path + "src_vocab.json"
        self.tgt_train_set, self.tgt_dev_set = self.save_path + "tgt_train.json", self.save_path + "tgt_dev.json"
        self.tgt_test_set, self.tgt_vocab = self.save_path + "tgt_test.json", self.save_path + "tgt_vocab.json"
        self.src_word_weight = self.save_path + "src_word_weight.npz"
        self.tgt_word_weight = self.save_path + "tgt_word_weight.npz"
        self.char_weight = self.save_path + "char_weight.npz"
        self.src_wordvec_path = os.path.join(EMB_PATH, EMB_DICT[self.src_language].format(self.src_word_dim))
        self.src_wordvec = self.save_path + "src_wordvec.npz"
        self.tgt_wordvec_path = os.path.join(EMB_PATH, EMB_DICT[self.tgt_language].format(self.tgt_word_dim))
        self.tgt_wordvec = self.save_path + "tgt_wordvec.npz"
