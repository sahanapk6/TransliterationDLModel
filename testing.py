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







def main():
    model = torch.load(
        "cnnmodel.epoch_1", map_location=torch.device('cpu'))
    decode_fn = get_decode_fn(Decode.greedy, 128, 5)
    tester = Trainer()
    params = tester.params
    tester.model = model
    tester.params.src_layer = 6
    tester.params.trg_layer = 6
    tester.params.epochs = 2
    tester.load_data(params.dataset, params.train, params.dev, params.test)
    tester.setup_evalutator()
    results = tester.testdecodeTransformer(TEST, 20, "decode", decode_fn)
    if results:
        for result in results:
            tester.logger.info(
                f"TEST {result.long_desc} is {result.res} at epoch -1"
            )
        results = " ".join([f"{r.desc} {r.res}" for r in results])
        tester.logger.info(f'TEST {"test"} {results}')


if __name__ == "__main__":
    main()
