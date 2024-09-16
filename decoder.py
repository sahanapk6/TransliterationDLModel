from collections import namedtuple

import torch
import torch.nn.functional as F
import util
from dataloader import BOS_IDX, EOS_IDX, STEP_IDX
from transformer import Transformer, CNNSeq2SeqTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decode(util.NamedEnum):
    greedy = "greedy"
    beam = "beam"


class Decoder(object):
    def __init__(
        self,
        decoder_type,
        max_len=100,
        beam_size=5,
        trg_bos=BOS_IDX,
        trg_eos=EOS_IDX,
        skip_attn=True,
    ):
        self.type = decoder_type
        self.max_len = max_len
        self.beam_size = beam_size
        self.trg_bos = trg_bos
        self.trg_eos = trg_eos
        self.skip_attn = skip_attn

    def __call__(self, transducer, src_sentence, src_mask):
        if self.type == Decode.greedy:
            if isinstance(transducer, Transformer):
                decode_fn = decode_greedy_transformer
            elif isinstance(transducer, CNNSeq2SeqTransformer):
                decode_fn = decode_greedy_CNNtransformer
            else:
                decode_fn = decode_greedy_default

            output, attns = decode_fn(
                transducer,
                src_sentence,
                src_mask,
                max_len=self.max_len,
                trg_bos=self.trg_bos,
                trg_eos=self.trg_eos,
            )
        else:
            raise ValueError
        return_values = (output, None if self.skip_attn else attns)
        return return_values


def get_decode_fn(decode, max_len=100, beam_size=5):
    return Decoder(decode, max_len=max_len, beam_size=beam_size)


def decode_greedy_default(
    transducer, src_sentence, src_mask, max_len=100, trg_bos=BOS_IDX, trg_eos=EOS_IDX
):
    """
    src_sentence: [seq_len]
    """
    transducer.eval()
    enc_hs = transducer.encode(src_sentence)
    _, bs = src_mask.shape

    hidden = transducer.dec_rnn.get_init_hx(bs)
    input_ = torch.tensor([trg_bos] * bs, device=DEVICE)
    output = input_.view(1, bs)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    attns = []

    finished = None
    for _ in range(max_len):
        word_logprob, hidden, attn = transducer.decode_step(
            enc_hs, src_mask, input_, hidden
        )
        word = torch.max(word_logprob, dim=1)[1]
        attns.append(attn)

        input_ = transducer.dropout(transducer.trg_embed(word))
        output = torch.cat((output, word.view(1, bs)))

        if finished is None:
            finished = word == trg_eos
        else:
            finished = finished | (word == trg_eos)

        if finished.all().item():
            break

    return output, attns


def decode_greedy_transformer(
    transducer, src_sentence, src_mask, max_len=100, trg_bos=BOS_IDX, trg_eos=EOS_IDX
):
    """
    src_sentence: [seq_len]
    """
    assert isinstance(transducer, Transformer)
    transducer.eval()
    src_mask = (src_mask == 0).transpose(0, 1)
    enc_hs = transducer.encode(src_sentence, src_mask)

    _, bs = src_sentence.shape
    output = torch.tensor([trg_bos] * bs, device=DEVICE)
    output = output.view(1, bs)

    finished = None
    for _ in range(max_len):
        trg_mask = dummy_mask(output)
        trg_mask = (trg_mask == 0).transpose(0, 1)

        word_logprob = transducer.decode(enc_hs, src_mask, output, trg_mask)
        word_logprob = word_logprob[-1]

        word = torch.max(word_logprob, dim=1)[1]
        output = torch.cat((output, word.view(1, bs)))
        print("output final", output)
        print("word", word)

        if finished is None:
            finished = word == trg_eos
        else:
            finished = finished | (word == trg_eos)

        if finished.all().item():
            break
    return output, None


def dummy_mask(seq):
    """
    create dummy mask (all 1)
    """
    if isinstance(seq, tuple):
        seq = seq[0]
    return torch.ones_like(seq, dtype=torch.float)


def decode_greedy_CNNtransformer(
    transducer, src_sentence, src_mask, max_len=100, trg_bos=BOS_IDX, trg_eos=EOS_IDX
):
    """
    src_sentence: [seq_len]
    """
    assert isinstance(transducer, CNNSeq2SeqTransformer)
    transducer.eval()
    src_mask = (src_mask == 0).transpose(0, 1)
    enc_hs = transducer.encode(src_sentence)

    _, bs = src_sentence.shape
    output = torch.tensor([trg_bos] * bs, device=DEVICE)
    output = output.view(1, bs)

    finished = None
    for _ in range(max_len):
        trg_mask = dummy_mask(output)
        trg_mask = (trg_mask == 0).transpose(0, 1)

        # logits = transducer.decode(enc_hs, src_mask, output, trg_mask)
        # logits = logits[-1]
        # probs = F.softmax(logits, dim=-1)
        # word = torch.argmax(probs, dim=1)
        # output = torch.cat((output, word.view(1, bs)))

        word_logprob = transducer.decode(enc_hs, output)
        word_logprob = word_logprob[-1]

        word = torch.max(word_logprob, dim=1)[1]
        output = torch.cat((output, word.view(1, bs)))

        # print("output final", output)
        if finished is None:
            finished = word == trg_eos
        else:
            finished = finished | (word == trg_eos)

        if finished.all().item():
            break
    return output, None
