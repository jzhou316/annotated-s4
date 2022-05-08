import torch
import torchvision
import torchvision.transforms as transforms


def linearize_blocks(image: torch.Tensor, block_size=(8, 8)) -> torch.Tensor:
    """Linearize an image by first linearizing inside each small block, then linearizing on the block level."""
    if image.ndim == 3:
        # RGB, size (3, height/nrows, width/ncols)
        channels, nrows, ncols = image.size()
    elif image.ndim == 2:
        # grayscale
        # TODO
        channels = 1
        nrows, ncols = image.size()
        image = image.unsqueeze(0)
    else:
        raise ValueError

    rblocks, cblocks = (nrows + block_size[0] - 1) // block_size[0], (ncols + block_size[1] - 1) // block_size[1]
    flattened_blocks = []
    for i in range(rblocks):
        for j in range(cblocks):
            # flatten each block
            flattened_blocks.append(
                image[:,
                      i * block_size[0]:(i + 1) * block_size[0],
                      j * block_size[1]:(j + 1) * block_size[1]
                      ].reshape(channels, -1)
                )
    # flatten all blocks
    blockseq = torch.cat(flattened_blocks, dim=1)    # of size (channel, seq_length)

    return blockseq


# ### CIFAR-10 Classification
# **Task**: Predict CIFAR-10 class given sequence model over pixels (32 x 32 x 3 RGB image => 10 classes).
#       NOTE the sequences are linearized in a special way to mimic the JPEG DCT ordering
#            - sequentialize 8 x 8 block first in row scanning order (could also try zigzag)
#            - sequentialize the blocks in row scanning order
def create_cifar_blockseq_classification_dataset(bsz=128):
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
# **Task**: Predict CIFAR-10 class given sequence model over pixels (32 x 32 x 3 RGB image => 10 classes).
#           No normalization.
def create_cifar_nonorm_classification_dataset(bsz=128):
    print("[*] Generating CIFAR-10 with no normalization (0-255) Classification Dataset")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 32 * 32, 10, 3
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            # ),
            transforms.Lambda(
                lambda x: (x * 256).int(),
            ),
            transforms.Lambda(lambda x: x.view(3, 1024).t()),
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
