import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from tqdm import tqdm

from jpeg_baseline_decoder.check_diff_jpeg import (jpg_bytes_from_file,
                                                   load_jpg2img,
                                                   compress_jpg, compress_jpg_torch, compress_jpg_simplejpeg)
from jpeg_baseline_decoder.jpeg_decoder import JPEG


def jpeg_huffman_decoding_stream(img: np.ndarray, quality: int):
    """Get the Huffman decoding data stream from JPEG compression bits.

    Input image is grayscale, of size (height, width) or RGB, of size (height, width, 3).

    First encode the image into JPEG, as we currently only support 4:4:4 subsampling (no chroma subsampling).
    Then decode the JPEG to extract the Huffman decoded data stream.
    """
    # first encode to JPEG
    jpeg_bytes = compress_jpg_simplejpeg(img, quality, colorsubsampling='444')
    # NOTE `colorsubsampling` does not matter for grayscale, so other encoding functions can be used. But for RGB,
    #      simplejpeg should be used as it can specify the subsampling rate

    # decode to get the Huffman decoded data stream
    jpeg = JPEG(jpeg_bytes=jpeg_bytes, verbose=False)
    decoded_huffman_stream = jpeg.get_decoded_huffman_stream()

    # dtype('float64'), shape (stream_len,)
    return np.array(decoded_huffman_stream)


def jpeg_huffman_decoding_wzeros_stream(img: np.ndarray, quality: int):
    """Get the Huffman decoding data stream from JPEG compression bits, with zero-run lengths recovered with actual zeros.

    Input image is grayscale, of size (height, width) or RGB, of size (height, width, 3).

    First encode the image into JPEG, as we currently only support 4:4:4 subsampling (no chroma subsampling).
    Then decode the JPEG to extract the Huffman decoded data stream.
    """
    # first encode to JPEG
    jpeg_bytes = compress_jpg_simplejpeg(img, quality, colorsubsampling='444')
    # NOTE `colorsubsampling` does not matter for grayscale, so other encoding functions can be used. But for RGB,
    #      simplejpeg should be used as it can specify the subsampling rate

    # decode to get the Huffman decoded data stream
    jpeg = JPEG(jpeg_bytes=jpeg_bytes, verbose=False)
    decoded_huffman_wzeros_stream = jpeg.get_decoded_huffman_zero_recovered_stream()

    # dtype('float64'), shape (stream_len,)
    return np.array(decoded_huffman_wzeros_stream)


# **Task**: Predict MNIST class given sequence model over pixels transformed to Huffman decoded stream from
#           compressed JPEG codes (max_len_with_padding => 10 classes).
def create_mnist_jpeg_huffdec_stream_classification_dataset(bsz=128, quality=75):
    print("[*] Generating MNIST JPEG Huffman Decoding Stream Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 1070, 10, 1

    """
    Training images: 60000
    Testing images: 10000
        size: (1, 28, 28) grayscale

    When JPEG quality=75, the lengths of the JPEG Huffman decoded sequences are ([0, 25, 50, 75, 100] percentiles):
    - train [48, 542, 616, 686, 1070]
    - test [48, 542, 616, 686, 1020]

    We pad to the right for all codes to the maximum length.
    """

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x * 256).byte().squeeze(0),    # grayscale
            ),
            transforms.Lambda(
                lambda x: torch.tensor(jpeg_huffman_decoding_stream(x.numpy(), quality=quality)),
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

    # preprocess the data as it is time comsuming
    train_data = []
    for i in tqdm(range(len(train)), desc=' Getting Huffman decoded data stream for train', unit=' images'):
        train_data.append(train[i][0])
    train_data = torch.stack(train_data)

    test_data = []
    for i in tqdm(range(len(test)), desc=' Getting Huffman decoded data stream for test', unit=' images'):
        test_data.append(test[i][0])
    test_data = torch.stack(test_data)

    train_processed = TensorDataset(train_data, train.targets)
    test_processed = TensorDataset(test_data, test.targets)

    # data_all = torch.Tensor()
    # for i in tqdm(range(len(train)), desc=' Getting Huffman decoded data stream for train', unit=' images'):
    #     data = torch.tensor(jpeg_huffman_decoding_stream(train[i][0].numpy(), quality=quality))
    #     data = torch.nn.functional.pad(data, (0, SEQ_LENGTH - len(data))).view(1, SEQ_LENGTH, 1)  # pad 0 at the end
    #     data_all = torch.cat([data_all, data], dim=0)
    # train.data = data_all
    # train.transform = None

    # data_all = torch.Tensor()
    # for i in tqdm(range(len(test)), desc=' Getting Huffman decoded data stream for test', unit=' images'):
    #     data = torch.tensor(jpeg_huffman_decoding_stream(test[i][0].numpy(), quality=quality))
    #     data = torch.nn.functional.pad(data, (0, SEQ_LENGTH - len(data))).view(1, SEQ_LENGTH, 1)  # pad 0 at the end
    #     data_all = torch.cat([data_all, data], dim=0)
    # test.data = data_all
    # test.transform = None

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train_processed, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_processed, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# **Task**: Predict MNIST class given sequence model over pixels transformed to Huffman decoded stream,
#           with zero-run length recovered to actual zeros from
#           compressed JPEG codes (32 x 32 = 1024 => 10 classes).
def create_mnist_jpeg_huffdec_wzeros_stream_classification_dataset(bsz=128, quality=75):
    print("[*] Generating MNIST JPEG Huffman Decoding with Zeros Stream Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 1024, 10, 1

    """
    Training images: 60000
    Testing images: 10000
        size: (1, 28, 28) grayscale

    The decoded Huffman stream with zeros recovered have size 32 x 32 = 1024, to cover all 8 x 8 blocks.
    """

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x * 256).byte().squeeze(0),    # grayscale
            ),
            transforms.Lambda(
                lambda x: torch.tensor(jpeg_huffman_decoding_wzeros_stream(x.numpy(), quality=quality)
                                       ).view(SEQ_LENGTH, 1),
            ),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # preprocess the data as it is time comsuming
    train_data = []
    for i in tqdm(range(len(train)), desc=' Getting Huffman decoded data stream for train', unit=' images'):
        train_data.append(train[i][0])
    train_data = torch.stack(train_data)

    test_data = []
    for i in tqdm(range(len(test)), desc=' Getting Huffman decoded data stream for test', unit=' images'):
        test_data.append(test[i][0])
    test_data = torch.stack(test_data)

    train_processed = TensorDataset(train_data, train.targets)
    test_processed = TensorDataset(test_data, test.targets)

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train_processed, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_processed, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### CIFAR-10 Classification
# **Task**: Predict CIFAR-10 class given sequence model over 32 x 32 pixels transformed to Huffman decoded stream from
#           compressed JPEG codes (max_len_with_padding => 10 classes).
def create_cifar_jpeg_huffdec_stream_classification_dataset(bsz=128, quality=75):
    print("[*] Generating CIFAR-10 JPEG Huffman Decoding Stream Classification Dataset")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = None, 10, 1    # NOTE SEQ_LENGTH to be decided in the data processing below

    """
    Training images: 50000
    Testing images: 10000
        size: (3, 32, 32) RGB

    When JPEG quality=75, the lengths of the JPEG Huffman decoded sequences are ([0, 25, 50, 75, 100] percentiles):
    - train [330, 972, 1114, 1274, 3850]
    - test [296, 972, 1112, 1270, 3510]
    - train + test [296, 972, 1114, 1272, 3850]

    We pad to the right for all codes to the maximum length.
    """

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x * 256).byte().permute(1, 2, 0),    # nrows, ncols, 3
            ),
            transforms.Lambda(
                lambda x: torch.tensor(jpeg_huffman_decoding_stream(x.numpy(), quality=quality)),
            ),
            # transforms.Lambda(
            #     lambda x: torch.nn.functional.pad(x, (0, SEQ_LENGTH - len(x))).view(SEQ_LENGTH, 1),  # pad 0 at the end
            # ),
        ]
    )

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )

    # preprocess the data as it is time comsuming
    train_data = []
    train_lens = []
    for i in tqdm(range(len(train)), desc=' Getting Huffman decoded data stream for train', unit=' images'):
        train_data.append(train[i][0])
        train_lens.append(len(train_data[-1]))

    test_data = []
    test_lens = []
    for i in tqdm(range(len(test)), desc=' Getting Huffman decoded data stream for test', unit=' images'):
        test_data.append(test[i][0])
        test_lens.append(len(test_data[-1]))

    # get the maximum sequence length
    SEQ_LENGTH = max(train_lens + test_lens)
    print('sequence length distribution train: percentiles [0, 25, 50, 75, 100]:')
    print(np.percentile(train_lens, [0, 25, 50, 75, 100]).astype(np.int).tolist())
    print('sequence length distribution test: percentiles [0, 25, 50, 75, 100]:')
    print(np.percentile(test_lens, [0, 25, 50, 75, 100]).astype(np.int).tolist())
    print('sequence length distribution train + test: percentiles [0, 25, 50, 75, 100]:')
    print(np.percentile(train_lens + test_lens, [0, 25, 50, 75, 100]).astype(np.int).tolist())

    # pad 0s at the end to the maximum length for all data
    print(f'padding 0s at the end of all sequences to the maximum length {SEQ_LENGTH}')
    train_data = [torch.nn.functional.pad(x, (0, SEQ_LENGTH - len(x))).view(SEQ_LENGTH, 1) for x in train_data]
    test_data = [torch.nn.functional.pad(x, (0, SEQ_LENGTH - len(x))).view(SEQ_LENGTH, 1) for x in test_data]

    train_data = torch.stack(train_data)
    test_data = torch.stack(test_data)

    train_processed = TensorDataset(train_data, torch.tensor(train.targets))
    test_processed = TensorDataset(test_data, torch.tensor(test.targets))
    # NOTE train.targets is a list, which is different from the MNIST data... such an inconsistency from torchvision...

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train_processed, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_processed, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### CIFAR-10 Classification
# **Task**: Predict CIFAR-10 class given sequence model over over pixels transformed to Huffman decoded stream,
#           with zero-run length recovered to actual zeros from
#           compressed JPEG codes (32 x 32 x 3 = 1024 x 3 => 10 classes).
# def create_cifar_jpeg_huffdec_wzeros_stream_classification_dataset(bsz=128, quality=75):
def create_cifar_jpeg_huffdec_wzeros_stream_classification_dataset(bsz=128, quality=100):
    print("[*] Generating CIFAR-10 JPEG Huffman Decoding with Zeros Stream Classification Dataset")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 32 * 32 * 3, 10, 1    # TODO try 3 channels as IN_DIM

    """
    Training images: 50000
    Testing images: 10000
        size: (3, 32, 32) grayscale

    The decoded Huffman stream with zeros recovered have size 32 x 32 x 3 = 1024 x 3 = 3072, to cover all 8 x 8 blocks.
    """

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x * 256).byte().permute(1, 2, 0),    # nrows, ncols, 3
            ),
            transforms.Lambda(
                lambda x: torch.tensor(jpeg_huffman_decoding_wzeros_stream(x.numpy(), quality=quality)
                                       ).view(SEQ_LENGTH, 1),
            ),
        ]
    )

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )

    # preprocess the data as it is time comsuming
    train_data = []
    for i in tqdm(range(len(train)), desc=' Getting Huffman decoded data stream for train', unit=' images'):
        train_data.append(train[i][0])
    train_data = torch.stack(train_data)

    test_data = []
    for i in tqdm(range(len(test)), desc=' Getting Huffman decoded data stream for test', unit=' images'):
        test_data.append(test[i][0])
    test_data = torch.stack(test_data)

    train_processed = TensorDataset(train_data, torch.tensor(train.targets))
    test_processed = TensorDataset(test_data, torch.tensor(test.targets))
    # NOTE train.targets is a list, which is different from the MNIST data... such an inconsistency from torchvision...

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train_processed, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_processed, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM
