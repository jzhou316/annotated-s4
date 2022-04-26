import numpy as np
from typing import List
import torch
import torchvision
import torchvision.transforms as transforms
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
        # fixed at subsampling 4:2:2
        # NOTE currently doing nothing
        ...
    else:
        # grayscale, cb and cr are all zeros
        assert dct_cb.sum() == 0 and dct_cr.sum() == 0

    return dct_y, dct_cb, dct_cr


def unblock_dct_linearize(dct_blocks: np.ndarray, block_size=(8, 8)):
    """convert the block structured linearized DCT coefficients (output from `jpeg2dct.loads`) to a flatten sequence,
    by linearizing for the whole image.
    """
    rblocks, cblocks, nvalues = dct_blocks.shape
    assert nvalues == block_size[0] * block_size[1]    # 64

    dct_flatten = np.empty(0, dtype=np.int16)    # type of the `dct_blocks``
    for i in range(rblocks):
        for j in range(cblocks):
            dct_flatten = np.concatenate([dct_flatten, dct_blocks[i, j]])

    return dct_flatten


def unblock_dct_linearize_ycbcr422(dct_blocks_y: np.ndarray, dct_blocks_cb: np.ndarray, dct_blocks_cr: np.ndarray,
                                   block_size=(8, 8)):
    """Convert the block structured linearized DCT coefficients (output from `jpeg2dct.loads`) to a flatten sequence,
    by linearing for the whole image.
    Deals with the RGB image, with Y Cb Cr components in the decoded DCT coefficients, with chroma subsampling 4:2:2.
    The shapes of these components for an original image of size 32 x 32 is:
        Y - 4 x 4 x 64, Cb - 2 x 2 x 64, Cr - 2 x 2 x 64
    due to chroma subsampling 4:2:2. Here we just simply linearize the blocks to concatenate Y Cb Cr for 2 x 2 blocks,
    followed by Y Cb Cr of another 2 x 2 blocks, and so on.
    """
    rblocks_y, cblocks_y, nvalues = dct_blocks_y.shape
    assert nvalues == block_size[0] * block_size[1]

    rblocks_c, cblocks_c, nvalues = dct_blocks_cb.shape
    assert rblocks_c * 2 == rblocks_y, 'Only deals with chorma subsampling 4:2:2'
    assert cblocks_c * 2 == cblocks_y, 'Only deals with chorma subsampling 4:2:2'

    dct_flatten = np.empty(0, dtype=np.int16)    # type of the `dct_blocks`
    for i in range(rblocks_c):
        for j in range(cblocks_c):
            # 4 Y blocks, for one Cb block and one Cr block
            for x in [i * 2, i * 2 + 1]:
                for y in [j * 2, j * 2 + 1]:
                    dct_flatten = np.concatenate([dct_flatten, dct_blocks_y[x, y]])
            # Cb block
            dct_flatten = np.concatenate([dct_flatten, dct_blocks_cb[i, j]])
            # Cr block
            dct_flatten = np.concatenate([dct_flatten, dct_blocks_cr[i, j]])

    return dct_flatten


def unblock_dct_linearize_ycbcr422_upsampling(
        dct_blocks_y: np.ndarray, dct_blocks_cb: np.ndarray, dct_blocks_cr: np.ndarray,
        block_size=(8, 8)):
    """Convert the block structured linearized DCT coefficients (output from `jpeg2dct.loads`) to a flatten sequence,
    by linearing for the whole image.
    Deals with the RGB image, with Y Cb Cr components in the decoded DCT coefficients, with chroma subsampling 4:2:2.
    The shapes of these components for an original image of size 32 x 32 is:
        Y - 4 x 4 x 64, Cb - 2 x 2 x 64, Cr - 2 x 2 x 64
    due to chroma subsampling 4:2:2. Here we just simply linearize the blocks to concatenate Y Cb Cr for 2 x 2 blocks,
    followed by Y Cb Cr of another 2 x 2 blocks, and so on.

    Upsample the Cb and Cr components in both width and height by 2.
    """
    rblocks_y, cblocks_y, nvalues = dct_blocks_y.shape
    assert nvalues == block_size[0] * block_size[1]

    rblocks_c, cblocks_c, nvalues = dct_blocks_cb.shape
    assert rblocks_c * 2 == rblocks_y, 'Only deals with chorma subsampling 4:2:2'
    assert cblocks_c * 2 == cblocks_y, 'Only deals with chorma subsampling 4:2:2'

    dct_blocks_cb_2d = np.zeros((rblocks_c * block_size[0], cblocks_c * block_size[1]))
    dct_blocks_cr_2d = np.zeros((rblocks_c * block_size[0], cblocks_c * block_size[1]))
    for i in range(rblocks_c):
        for j in range(cblocks_c):
            dct_blocks_cb_2d[i * block_size[0]:(i + 1) * block_size[0], j * block_size[1]:(j + 1) * block_size[1]] = \
                dct_blocks_cb[i, j].reshape(*block_size)
            dct_blocks_cr_2d[i * block_size[0]:(i + 1) * block_size[0], j * block_size[1]:(j + 1) * block_size[1]] = \
                dct_blocks_cr[i, j].reshape(*block_size)

    m = torch.nn.Upsample(scale_factor=2, mode='nearest')
    dct_blocks_cb_upsampled = m(torch.tensor(dct_blocks_cb, dtype=torch.float).view(
        1, 1, dct_blocks_cb_2d.shape[0], dct_blocks_cb_2d.shape[1])).squeeze(0).squeeze(0)
    dct_blocks_cr_upsampled = m(torch.tensor(dct_blocks_cr, dtype=torch.float).view(
        1, 1, dct_blocks_cr_2d.shape[0], dct_blocks_cr_2d.shape[1])).squeeze(0).squeeze(0)
    # outputs are torch.Tensor

    dct_flatten_y = np.empty(0, dtype=np.int16)    # type of the `dct_blocks`
    dct_flatten_cb = np.empty(0, dtype=np.int16)
    dct_flatten_cr = np.empty(0, dtype=np.int16)

    for x in range(rblocks_y):
        for y in range(cblocks_y):
            dct_flatten_y = np.concatenate([dct_flatten_y, dct_blocks_y[x, y]])
            # take out the linearized block component: row scan order
            dct_block_cb_flatten = dct_blocks_cb_upsampled[x * block_size[0]:(x + 1) * block_size[0],
                                                           y * block_size[1]:(y + 1) * block_size[1]
                                                           ].reshape(-1).numpy()
            dct_block_cr_flatten = dct_blocks_cr_upsampled[x * block_size[0]:(x + 1) * block_size[0],
                                                           y * block_size[1]:(y + 1) * block_size[1]
                                                           ].reshape(-1).numpy()
            dct_flatten_cb = np.concatenate([dct_flatten_cb, dct_block_cb_flatten])
            dct_flatten_cr = np.concatenate([dct_flatten_cr, dct_block_cr_flatten])

    dct_flatten_stack = np.stack([dct_flatten_y, dct_flatten_cb, dct_flatten_cr]).T
    return dct_flatten_stack


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
                lambda x: img2jpg2dct_blocks(x.numpy(), flag=0, quality=quality)[
                    0],    # grayscale, Cb and Cr are both 0s
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


# ### CIFAR-10 Classification
# **Task**: Predict CIFAR-10 class given sequence model over pixels transformed to DCT coefficients from
#           compressed JPEG codes (32 x 32 x 3 YCbCr DCT coefficients => 10 classes).
def create_cifar_jpeg_dct_classification_dataset(bsz=128, quality=75):
    print("[*] Generating CIFAR-10 JPEG DCT Coefficients Classification Dataset")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 32 * 32 + 256 + 256, 10, 1

    """
    Training images: 50000
    Testing images: 10000
        size: (3, 32, 32) RGB

    The decoded DCT coefficients have size (4, 4, 64) for the Y component, as each of the (4, 4) block is a 8 x 8 block;
    (2, 2, 64) for the Cb/Cr component, and each of the (2, 2) block covers (8 x 2) x (8 x 2) pixels, due to 4:2:2
    chroma subsampling.
    so the recovered DCT image is 32 x 32 + 16 * 16 + 16 * 16.
    """

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x * 256).byte(),
            ),
            transforms.Lambda(
                lambda x: img2jpg2dct_blocks(x.permute(1, 2, 0).numpy(), flag=1, quality=quality),
                # Y, Cb, Cr, with 4:2:2 chroma subsampling
            ),
            transforms.Lambda(
                lambda x: unblock_dct_linearize_ycbcr422(*x,).reshape(SEQ_LENGTH, 1)
            ),
            torch.tensor,
        ]
    )

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.CIFAR10(
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


# ### CIFAR-10 Classification
# **Task**: Predict CIFAR-10 class given sequence model over pixels transformed to DCT coefficients from
#           compressed JPEG codes (32 x 32 x 3 YCbCr DCT coefficients => 10 classes).
def create_cifar_jpeg_dct_stack_classification_dataset(bsz=128, quality=75):
    print("[*] Generating CIFAR-10 JPEG DCT Coefficients Stacked Classification Dataset")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 32 * 32, 10, 3

    """
    Training images: 50000
    Testing images: 10000
        size: (3, 32, 32) RGB

    The decoded DCT coefficients have size (4, 4, 64) for the Y component, as each of the (4, 4) block is a 8 x 8 block;
    (2, 2, 64) for the Cb/Cr component, and each of the (2, 2) block covers (8 x 2) x (8 x 2) pixels, due to 4:2:2
    chroma subsampling.
    We do upsampling of scale 2 for the Cb and Cr component, and stack the Y Cb Cr in the channel dimension to formulate
    input sequences of size 1024 x 3.
    """

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x * 256).byte(),
            ),
            transforms.Lambda(
                lambda x: img2jpg2dct_blocks(x.permute(1, 2, 0).numpy(), flag=1, quality=quality),
                # Y, Cb, Cr, with 4:2:2 chroma subsampling
            ),
            transforms.Lambda(
                lambda x: unblock_dct_linearize_ycbcr422_upsampling(*x,).reshape(SEQ_LENGTH, 3)
            ),
            torch.tensor,
        ]
    )

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.CIFAR10(
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
