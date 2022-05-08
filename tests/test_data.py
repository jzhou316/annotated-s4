import torch
import torchvision
import torchvision.transforms as transforms


def test_data_cifar_blockseq():
    from s4.data_cifar_blockseq import linearize_blocks

    print("[*] Generating CIFAR-10 Block Linearized Sequence Classification Dataset")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 32 * 32, 10, 3
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
            transforms.Lambda(lambda x: linearize_blocks(x).t()),
        ]
    )

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=tf
    )

    assert train[0][0].size() == (SEQ_LENGTH, IN_DIM)

    # breakpoint()

    from IPython import embed; embed()
    # NOTE with pytest, must run with `-s` or `--capture=no` flag
    # https://stackoverflow.com/questions/24617397/how-to-print-to-console-in-pytest

    return
