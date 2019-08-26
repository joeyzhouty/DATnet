import os
import re
import codecs
import ujson
import emoji
import numpy as np
from tqdm import tqdm
from collections import Counter

np.random.seed(12345)
emoji_unicode = {v: k for k, v in emoji.EMOJI_UNICODE.items()}
glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
PAD = "<PAD>"
UNK = "<UNK>"


def write_json(filename, dataset):
    with codecs.open(filename, mode="w", encoding="utf-8") as f:
        ujson.dump(dataset, f)


def iob_to_iobes(labels):
    """IOB -> IOBES"""
    iob_to_iob2(labels)
    new_tags = []
    for i, tag in enumerate(labels):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(labels) and labels[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(labels) and labels[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iob_to_iob2(labels):
    """Check that tags have a valid IOB format. Tags in IOB1 format are converted to IOB2."""
    for i, tag in enumerate(labels):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or labels[i - 1] == 'O':  # conversion IOB1 to IOB2
            labels[i] = 'B' + tag[1:]
        elif labels[i - 1][1:] == tag[1:]:
            continue
        else:
            labels[i] = 'B' + tag[1:]
    return True


def word_convert(word, lowercase=True, char_lowercase=False):
    if char_lowercase:
        char = [c for c in word.lower()]
    else:
        char = [c for c in word]
    if lowercase:
        word = word.lower()
    return word, char


def remove_emoji(line):
    line = "".join(char for char in line if char not in emoji_unicode)
    try:
        pattern = re.compile(u"([\U00002600-\U000027BF])|([\U0001F1E0-\U0001F6FF])")
    except re.error:
        pattern = re.compile(u"([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|"
                             u"([\uD83D][\uDE80-\uDEFF])")
    return pattern.sub(r'', line)


def process_wnut_token(line):
    line = remove_emoji(line)
    line = line.lstrip().rstrip().split("\t")
    if len(line) != 2:
        return None, None
    word, label = line[0], line[1]
    if word.startswith("@") or word.startswith("https://") or word.startswith("http://"):
        return None, None
    if word in ["&gt;", "&quot;", "&lt;", ":D", ";)", ":)", "-_-", "=D", ":'", "-__-", ":P", ":p", "RT", ":-)", ";-)",
                ":(", ":/"]:
        return None, None
    if "&amp;" in word:
        word = word.replace("&amp;", "&")
    if word in ["/", "<"] and label == "O":
        return None, None
    if len(word) == 0:
        return None, None
    return word, label


def process_token(line):
    word, *_, label = line.split(" ")
    if "page=http" in word or "http" in word:
        return None, None
    return word, label


def raw_dataset_iter(filename, language, lowercase=True, char_lowercase=False):
    with codecs.open(filename, mode="r", encoding="utf-8") as f:
        words, chars, labels = [], [], []
        for line in f:
            line = line.lstrip().rstrip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(words) != 0:
                    yield words, chars, labels
                    words, chars, labels = [], [], []
            else:
                if "wnut" in language:
                    word, label = process_wnut_token(line)
                else:
                    word, label = process_token(line)
                if word is None or label is None:
                    continue
                word, char = word_convert(word, lowercase=lowercase, char_lowercase=char_lowercase)
                words.append(word)
                chars.append(char)
                labels.append(label)
        if len(words) != 0:
            yield words, chars, labels


def load_dataset(filename, iobes, language, lowercase=True, char_lowercase=False):
    dataset = []
    for words, chars, labels in raw_dataset_iter(filename, language, lowercase, char_lowercase):
        if iobes:
            labels = iob_to_iobes(labels)
        dataset.append({"words": words, "chars": chars, "labels": labels})
    return dataset


def load_emb_vocab(data_path, language, dim):
    vocab = list()
    with codecs.open(data_path, mode="r", encoding="utf-8") as f:
        if "glove" in data_path:
            total = glove_sizes[data_path.split(".")[-3]]
        else:
            total = int(f.readline().lstrip().rstrip().split(" ")[0])
        for line in tqdm(f, total=total, desc="Load {} embedding vocabulary".format(language)):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2:
                continue
            if len(line) != dim + 1:
                continue
            word = line[0]
            vocab.append(word)
    return vocab


def filter_emb(word_dict, data_path, language, dim):
    vectors = np.zeros([len(word_dict), dim])
    with codecs.open(data_path, mode="r", encoding="utf-8") as f:
        if "glove" in data_path:
            total = glove_sizes[data_path.split(".")[-3]]
        else:
            total = int(f.readline().lstrip().rstrip().split(" ")[0])
        for line in tqdm(f, total=total, desc="Load {} embedding vectors".format(language)):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2:
                continue
            if len(line) != dim + 1:
                continue
            word = line[0]
            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_idx = word_dict[word]
                vectors[word_idx] = np.asarray(vector)
    return vectors


def build_token_counters(datasets):
    word_counter = Counter()
    char_counter = Counter()
    label_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            for word in record["words"]:
                word_counter[word] += 1
            for char in record["chars"]:
                for c in char:
                    char_counter[c] += 1
            for label in record["labels"]:
                label_counter[label] += 1
    return word_counter, char_counter, label_counter


def build_dataset(data, word_dict, char_dict, label_dict):
    dataset = []
    for record in data:
        chars_list = []
        words = []
        for word in record["words"]:
            words.append(word_dict[word] if word in word_dict else word_dict[UNK])
        for char in record["chars"]:
            chars = [char_dict[c] if c in char_dict else char_dict[UNK] for c in char]
            chars_list.append(chars)
        labels = [label_dict[label] for label in record["labels"]]
        dataset.append({"words": words, "chars": chars_list, "labels": labels})
    return dataset


def read_data_and_vocab(train_file, dev_file, test_file, language, config):
    train_data = load_dataset(train_file, iobes=config.iobes, language=language, lowercase=config.word_lowercase,
                              char_lowercase=config.char_lowercase)
    dev_data = load_dataset(dev_file, iobes=config.iobes, language=language, lowercase=config.word_lowercase,
                            char_lowercase=config.char_lowercase)
    test_data = load_dataset(test_file, iobes=config.iobes, language=language, lowercase=config.word_lowercase,
                             char_lowercase=config.char_lowercase)
    word_counter, char_counter, label_counter = build_token_counters([train_data, dev_data, test_data])
    return train_data, dev_data, test_data, word_counter, char_counter, label_counter


def write_to_jsons(datasets, files, save_path):
    for dataset, file in zip(datasets, files):
        write_json(os.path.join(save_path, file), dataset)


def process_word_counter(word_counter, wordvec_path, wordvec, language, word_dim, word_weight):
    word_vocab = [word for word, _ in word_counter.most_common()]
    if wordvec is not None:
        emb_vocab = load_emb_vocab(wordvec_path, language=language, dim=word_dim)
        word_vocab = list(set(word_vocab) & set(emb_vocab))
        tmp_word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
        vectors = filter_emb(tmp_word_dict, wordvec_path, language, word_dim)
        np.savez_compressed(wordvec, embeddings=np.asarray(vectors))
    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    # create word weight for adversarial training
    word_count = dict()
    for word, count in word_counter.most_common():
        if word in word_dict:
            word_count[word] = word_count.get(word, 0) + count
        else:
            word_count[UNK] = word_count.get(UNK, 0) + count
    sum_word_count = float(sum(list(word_count.values())))
    word_weight_vec = [float(word_count[word]) / sum_word_count for word in word_vocab[1:]]
    np.savez_compressed(word_weight, embeddings=np.asarray(word_weight_vec))
    return word_dict


def process_char_counter(char_counter, threshold, char_weight):
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= threshold]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    # create character weight for adversarial training
    char_count = dict()
    for char, count in char_counter.most_common():
        if char in char_dict:
            char_count[char] = char_count.get(char, 0) + count
        else:
            char_count[UNK] = char_count.get(UNK, 0) + count
    sum_char_count = float(sum(list(char_count.values())))
    char_weight_vec = [float(char_count[char]) / sum_char_count for char in char_vocab[1:]]  # exclude PAD
    np.savez_compressed(char_weight, embeddings=np.asarray(char_weight_vec))
    return char_dict


def process_base(config):
    train_data, dev_data, test_data, word_counter, char_counter, label_counter = read_data_and_vocab(
        config.train_file, config.dev_file, config.test_file, language=config.language, config=config)
    # create save path
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    # build word vocab
    word_dict = process_word_counter(word_counter, config.wordvec_path, config.wordvec, config.language,
                                     config.word_dim, config.word_weight)
    # build char vocab
    char_dict = process_char_counter(char_counter, config.threshold, config.char_weight)
    # build label vocab
    label_vocab = ["O"] + [label for label, _ in label_counter.most_common() if label != "O"]
    label_dict = dict([(label, idx) for idx, label in enumerate(label_vocab)])
    # create indices dataset
    if config.dev_for_train:
        train_data = train_data + dev_data
    train_set = build_dataset(train_data, word_dict, char_dict, label_dict)
    dev_set = build_dataset(dev_data, word_dict, char_dict, label_dict)
    test_set = build_dataset(test_data, word_dict, char_dict, label_dict)
    vocab = {"word_dict": word_dict, "char_dict": char_dict, "label_dict": label_dict}
    # write to json
    write_to_jsons([train_set, dev_set, test_set, vocab], ["train.json", "dev.json", "test.json", "vocab.json"],
                   config.save_path)


def process_transfer(config):
    s_train_data, s_dev_data, s_test_data, s_word_counter, s_char_counter, s_label_counter = read_data_and_vocab(
        config.src_train_file, config.src_dev_file, config.src_test_file, language=config.src_language, config=config)
    t_train_data, t_dev_data, t_test_data, t_word_counter, t_char_counter, t_label_counter = read_data_and_vocab(
        config.tgt_train_file, config.tgt_dev_file, config.tgt_test_file, language=config.tgt_language, config=config)
    # create save path
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    # build word vocab
    if config.share_word:
        s_word_counter = s_word_counter + t_word_counter
        word_dict = process_word_counter(s_word_counter, config.src_wordvec_path, config.src_wordvec,
                                         config.src_language, config.src_word_dim, config.src_word_weight)
        src_word_dict, tgt_word_dict = word_dict.copy(), word_dict.copy()
    else:
        src_word_dict = process_word_counter(s_word_counter, config.src_wordvec_path, config.src_wordvec,
                                             config.src_language, config.src_word_dim, config.src_word_weight)
        tgt_word_dict = process_word_counter(t_word_counter, config.tgt_wordvec_path, config.tgt_wordvec,
                                             config.tgt_language, config.tgt_word_dim, config.tgt_word_weight)
    # build char vocab
    s_char_counter = s_char_counter + t_char_counter
    char_dict = process_char_counter(s_char_counter, config.threshold, config.char_weight)
    # build label vocab
    if config.share_label:
        s_label_counter = s_label_counter + t_label_counter
        label_vocab = ["O"] + [label for label, _ in s_label_counter.most_common() if label != "O"]
        label_dict = dict([(label, idx) for idx, label in enumerate(label_vocab)])
        src_label_dict = label_dict.copy()
        tgt_label_dict = label_dict.copy()
    else:
        src_label_vocab = ["O"] + [label for label, _ in s_label_counter.most_common() if label != "O"]
        src_label_dict = dict([(label, idx) for idx, label in enumerate(src_label_vocab)])
        tgt_label_vocab = ["O"] + [label for label, _ in t_label_counter.most_common() if label != "O"]
        tgt_label_dict = dict([(label, idx) for idx, label in enumerate(tgt_label_vocab)])
    # create indices dataset
    src_train_set = build_dataset(s_train_data, src_word_dict, char_dict, src_label_dict)
    src_dev_set = build_dataset(s_dev_data, src_word_dict, char_dict, src_label_dict)
    src_test_set = build_dataset(s_test_data, src_word_dict, char_dict, src_label_dict)
    src_vocab = {"word_dict": src_word_dict, "char_dict": char_dict, "label_dict": src_label_dict}
    if config.dev_for_train:
        t_train_data = t_train_data + t_dev_data
    tgt_train_set = build_dataset(t_train_data, tgt_word_dict, char_dict, tgt_label_dict)
    tgt_dev_set = build_dataset(t_dev_data, tgt_word_dict, char_dict, tgt_label_dict)
    tgt_test_set = build_dataset(t_test_data, tgt_word_dict, char_dict, tgt_label_dict)
    tgt_vocab = {"word_dict": tgt_word_dict, "char_dict": char_dict, "label_dict": tgt_label_dict}
    # write to json
    write_to_jsons([src_train_set, src_dev_set, src_test_set, src_vocab],
                   ["src_train.json", "src_dev.json", "src_test.json", "src_vocab.json"],
                   config.save_path)
    write_to_jsons([tgt_train_set, tgt_dev_set, tgt_test_set, tgt_vocab],
                   ["tgt_train.json", "tgt_dev.json", "tgt_test.json", "tgt_vocab.json"],
                   config.save_path)
