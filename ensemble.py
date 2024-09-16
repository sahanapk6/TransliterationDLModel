
from main import Trainer, get_decode_fn
from decoder import Decode
import numpy as np
from tqdm import tqdm
from math import ceil
import torch


TRAIN = "train"
DEV = "dev"
TEST = "test"
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


def ensemble():
    cnnmodel = torch.load(
        "cnnmodel.epoch_1", map_location=torch.device('cpu'))
    decode_fn = get_decode_fn(Decode.greedy, 128, 5)
    tester = Trainer()
    params = tester.params
    tester.model = cnnmodel
    tester.params.src_layer = 6
    tester.params.trg_layer = 6
    tester.params.epochs = 2
    tester.load_data(params.dataset, params.train, params.dev, params.test)
    tester.setup_evalutator()
    distscnn, predictedcnn = tester.CNNresults(TEST, 20, "decode", decode_fn)

    transformermodel = torch.load("transformer-kan-tel.epoch_1")
    decode_fn = get_decode_fn(Decode.greedy, 128, 5)
    tester = Trainer()
    params = tester.params
    tester.model = transformermodel
    tester.params.src_layer = 6
    tester.params.trg_layer = 6
    tester.params.epochs = 2
    tester.load_data(params.dataset, params.train, params.dev, params.test)
    tester.setup_evalutator()
    diststransformer, predictedtransformer = tester.Transformerresults(
        TEST, 20, "decode", decode_fn)

    for i in range(len(diststransformer)):
        if diststransformer[i] > distscnn[i]:
            print("predicted", predictedcnn[i])
        else:
            print("predicted", predictedtransformer[i])


if __name__ == "__main__":
    ensemble()
