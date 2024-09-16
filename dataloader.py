

import xml.etree.ElementTree
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# from align import Aligner

BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"
ALIGN = "<a>"
STEP = "<step>"
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
STEP_IDX = 4


class Dataloader(object):
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


class Seq2SeqDataLoader(Dataloader):
    def __init__(
        self,
        train_file: List[str],
        dev_file: List[str],
        test_file: Optional[List[str]] = None,
        shuffle=False,
    ):
        super().__init__()
        self.train_file = train_file[0] if len(train_file) == 1 else train_file
        self.dev_file = dev_file[0] if len(dev_file) == 1 else dev_file
        self.test_file = (
            test_file[0] if test_file and len(test_file) == 1 else test_file
        )
        self.shuffle = shuffle
        self.batch_data: Dict[str, List] = dict()
        self.nb_train, self.nb_dev, self.nb_test = 0, 0, 0
        self.nb_attr = 0
        self.source, self.target = self.build_vocab()
        self.source_vocab_size = len(self.source)
        self.target_vocab_size = len(self.target)
        self.attr_c2i: Optional[Dict]
        if self.nb_attr > 0:
            self.source_c2i = {c: i for i, c in enumerate(
                self.source[: -self.nb_attr])}
            self.attr_c2i = {
                c: i + len(self.source_c2i)
                for i, c in enumerate(self.source[-self.nb_attr:])
            }
        else:
            self.source_c2i = {c: i for i, c in enumerate(self.source)}
            self.attr_c2i = None
        self.target_c2i = {c: i for i, c in enumerate(self.target)}
        self.sanity_check()

    def sanity_check(self):
        assert self.source[PAD_IDX] == PAD
        assert self.target[PAD_IDX] == PAD
        assert self.source[BOS_IDX] == BOS
        assert self.target[BOS_IDX] == BOS
        assert self.source[EOS_IDX] == EOS
        assert self.target[EOS_IDX] == EOS
        assert self.source[UNK_IDX] == UNK
        assert self.target[UNK_IDX] == UNK

    def build_vocab(self):
        src_set, trg_set = set(), set()
        self.nb_train = 0
        for src, trg in self.read_file(self.train_file):
            self.nb_train += 1
            src_set.update(src)
            trg_set.update(trg)
        self.nb_dev = sum([1 for _ in self.read_file(self.dev_file)])
        if self.test_file is not None:
            self.nb_test = sum([1 for _ in self.read_file(self.test_file)])
        source = [PAD, BOS, EOS, UNK] + sorted(list(src_set))
        target = [PAD, BOS, EOS, UNK] + sorted(list(trg_set))
        return source, target

    def read_file(self, file):
        raise NotImplementedError

    def _file_identifier(self, file):
        return file

    def list_to_tensor(self, lst: List[List[int]], max_seq_len=None):
        max_len = max([len(x) for x in lst])
        if max_seq_len is not None:
            max_len = min(max_len, max_seq_len)
        data = torch.zeros((max_len, len(lst)), dtype=torch.long)
        for i, seq in tqdm(enumerate(lst), desc="build tensor"):
            data[: len(seq), i] = torch.tensor(seq)
        mask = (data > 0).float()
        return data, mask

    def _batch_sample(self, batch_size, file, shuffle):
        key = self._file_identifier(file)
        if key not in self.batch_data:
            lst = list()
            for src, trg in tqdm(self._iter_helper(file), desc="read file"):
                lst.append((src, trg))
            src_data, src_mask = self.list_to_tensor([src for src, _ in lst])
            trg_data, trg_mask = self.list_to_tensor([trg for _, trg in lst])
            self.batch_data[key] = (src_data, src_mask, trg_data, trg_mask)

        src_data, src_mask, trg_data, trg_mask = self.batch_data[key]
        nb_example = len(src_data[0])
        if shuffle:
            idx = np.random.permutation(nb_example)
        else:
            idx = np.arange(nb_example)
        for start in range(0, nb_example, batch_size):
            idx_ = idx[start: start + batch_size]
            src_mask_b = src_mask[:, idx_]
            trg_mask_b = trg_mask[:, idx_]
            src_len = int(src_mask_b.sum(dim=0).max().item())
            trg_len = int(trg_mask_b.sum(dim=0).max().item())
            src_data_b = src_data[:src_len, idx_].to(self.device)
            trg_data_b = trg_data[:trg_len, idx_].to(self.device)
            src_mask_b = src_mask_b[:src_len].to(self.device)
            trg_mask_b = trg_mask_b[:trg_len].to(self.device)
            yield (src_data_b, src_mask_b, trg_data_b, trg_mask_b)

    def train_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.train_file, shuffle=self.shuffle)

    def dev_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.dev_file, shuffle=False)

    def test_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.test_file, shuffle=False)

    def encode_source(self, sent):
        if sent[0] != BOS:
            sent = [BOS] + sent
        if sent[-1] != EOS:
            sent = sent + [EOS]
        seq_len = len(sent)
        s = []
        for x in sent:
            if x in self.source_c2i:
                s.append(self.source_c2i[x])
            else:
                s.append(self.attr_c2i[x])
        return torch.tensor(s, device=self.device).view(seq_len, 1)

    def decode_source(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.source[x] for x in sent]

    def decode_target(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.target[x] for x in sent]

    def _sample(self, file):
        for src, trg in self._iter_helper(file):
            yield (
                torch.tensor(src, device=self.device).view(len(src), 1),
                torch.tensor(trg, device=self.device).view(len(trg), 1),
            )

    def train_sample(self):
        yield from self._sample(self.train_file)

    def dev_sample(self):
        yield from self._sample(self.dev_file)

    def test_sample(self):
        yield from self._sample(self.test_file)

    def _iter_helper(self, file):
        for source, target in self.read_file(file):
            src = [self.source_c2i[BOS]]
            for s in source:
                src.append(self.source_c2i.get(s, UNK_IDX))
            src.append(self.source_c2i[EOS])
            trg = [self.target_c2i[BOS]]
            for t in target:
                trg.append(self.target_c2i.get(t, UNK_IDX))
            trg.append(self.target_c2i[EOS])
            yield src, trg


class Transliteration(Seq2SeqDataLoader):
    def read_file(self, file):
        root = xml.etree.ElementTree.parse(file).getroot()
        for names in root.findall("row"):
            names = [n.text for n in names]
            src, trgs = names[2], names[3:]
            for trg in trgs:
                if trg is None:
                    trg = "Null"
                yield list(src), list(trg)


# class Transliteration(Seq2SeqDataLoader):
#     def read_file(self, file):
#         # if type(file) == list:
#         #     for f in file:
#         #         root = xml.etree.ElementTree.parse(f).getroot()
#         #         for names in root.findall("Name"):
#         #             names = [n.text for n in names]
#         #             src, trgs = names[0], names[1:]
#         #             for trg in trgs:
#         #                 if trg is None:
#         #                     trg = "Null"
#         #                 yield list(src), list(trg)
#         # else:
#         root = xml.etree.ElementTree.parse(file).getroot()
#         for names in root.findall("Name"):
#             names = [n.text for n in names]
#             src, trgs = names[0], names[1:]
#             for trg in trgs:
#                 if trg is None:
#                     trg = "Null"
#                 yield list(src), list(trg)

#     def read_csv_file(self, file):
#         data = pd.read_csv(file)
#         for i in range(len(data)):
#             row = data.iloc[[i]]
#             print(row[2], row[3:])
#             src, trgs = i[2], i[3:]
#             for trg in trgs:
#                 if trg is None:
#                     trg = "Null"
#                 yield list(src), list(trg)
