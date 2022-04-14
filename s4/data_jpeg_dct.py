import os
import jax
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm
import cv2
from jpeg2dct.numpy import load, loads


def load_jpg2img(path: str, flag: bool):
    img = cv2.imread(path, flag)
    # img is np.ndarray, of size (nrows, ncols, 3) if flag else (nrows, ncols)
    return img


def load_jpg2dct_blocks(path: str):
    # read from a file
    dct_y, dct_cb, dct_cr = load(path)

    # # OR alternative way
    # # read from in memory buffer
    # with open(path, 'rb') as src:
    #     buffer = src.read()
    # dct_y, dct_cb, dct_cr = loads(buffer)

    return dct_y, dct_cb, dct_cr


def img2jpg2dct_blocks(img: np.ndarray, flag: bool, quality: int = 75):
    if flag:
        # rgb, 3 channel
        assert img.ndim == 3 and img.shape[2] == 3
        nrows, ncols, _ = img.shape
    else:
        # grayscale, 1 channel
        assert img.ndim == 2
        nrows, ncols = img.shape

    # encode image to compressed jpg code
    retval, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    # retval is bool, buf is np.ndarray
    assert retval, 'failed to encode image'
    jpg_bytes = buf.tobytes()    # type `bytes`

    # partially decode the jpg bytes to DCT coefficients
    dct_y, dct_cb, dct_cr = loads(jpg_bytes)

    if flag:
        # rgb, 3 channel
        # TODO
        ...
    else:
        # grayscale, cb and cr are all zeros
        assert dct_cb.sum() == 0 and dct_cr.sum() == 0

    return dct_y, dct_cb, dct_cr


def unblock_dct_linearize(dct_blocks: np.ndarray, block_size=(8, 8)):
    """convert the block structured linearized DCT coefficients (output from `jpeg2dct.loads`) to a flatten sequence,
    by linearizing for the whole image.
    """
    rblocks, cblocks, nvalues  = dct_blocks.shape
    assert nvalues == block_size[0] * block_size[1]    # 64

    dct_flatten = np.empty(0, dtype=np.int16)    # type of the `dct_blocks``
    for i in range(rblocks):
        for j in range(cblocks):
            dct_flatten = np.concatenate([dct_flatten, dct_blocks[i, j]])

    return dct_flatten


# **Task**: Predict MNIST class given sequence model over pixels transformed to JPEG codes (784 pixels => 10 classes).
def create_mnist_jpeg_classification_dataset(bsz=128):
    print("[*] Generating MNIST JPEG Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 728, 10, 1

    """
    Training images: 60000
    Testing images: 10000
        size: (1, 28, 28) grayscale

    When JPEG quality=75, the lengths of the JPEG codes are ([0, 25, 50, 75, 100] percentiles):
    - train [344, 538, 566, 592, 728]
    - test [344, 539, 567, 592, 698]

    We pad to the right for all codes to the maximum length.
    """

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x * 256).byte()
            ),
            transforms.Lambda(
                lambda x: torchvision.io.encode_jpeg(x, quality=75),
            ),
            transforms.Lambda(
                lambda x: torch.nn.functional.pad(x, (0, SEQ_LENGTH - len(x))).view(SEQ_LENGTH, 1),  # pad 0 at the end
            ),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# **Task**: Predict MNIST class given sequence model over pixels transformed to DCT coefficients from
#           compressed JPEG codes (32 * 32 = 1024 coefficients => 10 classes).
def create_mnist_jpeg_dct_classification_dataset(bsz=128, quality=75):
    print("[*] Generating MNIST JPEG DCT Coefficients Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 1024, 10, 1

    """
    Training images: 60000
    Testing images: 10000
        size: (1, 28, 28) grayscale

    The decoded DCT coefficients have size (4, 4, 64) for the Y component, as each of the (4, 4) block is a 8 x 8 block,
    so the recovered DCT image is 32 x 32.
    """

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x * 256).byte().squeeze(0),
            ),
            transforms.Lambda(
                lambda x: img2jpg2dct_blocks(x.numpy(), flag=0, quality=quality)[0],    # grayscale, Cb and Cr are both 0s
            ),
            transforms.Lambda(
                lambda x: unblock_dct_linearize(x).reshape(SEQ_LENGTH, 1),
            ),
            torch.tensor,
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM
