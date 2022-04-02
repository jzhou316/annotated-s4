"""Extract hidden states on data using trained models."""
import argparse
import os
from functools import partial

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from tqdm import tqdm

from .s4 import BatchSeqModel, S4LayerInit
from .data_tata import create_tata_dataset
from .train_tata import cross_entropy_loss, compute_accuracy


checkpoint_dir = ('/cephfs_nese/TRANSFER/rjsingh/DDoS/DDoS/annotated-s4/'
                  'checkpoints/netflow/')

checkpoint_name = ('512_s4-d_model=128_dp0.05/'
                   'best_91')
seq_length = 512
bsz = 128
d_model = 128

checkpoint_name = ('s4-d_model=128-sl4096-dp0.05-bsz128/'
                   'best_99')
seq_length = 4096
bsz = 128
d_model = 128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir,
                        help='directory of the saved checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=checkpoint_name,
                        help='name of the saved checkpoint to be used')
    parser.add_argument(
        "--seq_length", type=int, default=seq_length, help="sequence length",
    )
    parser.add_argument("--bsz", type=int, default=bsz)

    # Model Parameters
    parser.add_argument("--d_model", type=int, default=d_model)
    parser.add_argument("--n_layers", type=int, default=4)

    # S4 Specific Parameters
    parser.add_argument("--ssm_n", type=int, default=64)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)

    # ========== Load data

    # Create dataset...
    trainloader, testloader, n_classes, seq_len, in_dim = create_tata_dataset(
        seq_length=args.seq_length,
        bsz=args.bsz,
    )

    # ========== load model

    model = S4LayerInit(N=64)
    model = partial(
        BatchSeqModel,
        layer=model,
        d_output=n_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        l_max=seq_len - 1,
    )
    rng = jax.random.PRNGKey(0)
    state = checkpoints.restore_checkpoint(checkpoint_path, None)
    model = model(training=False)

    # breakpoint()

    # ========== apply model to data
    losses, accuracies = [], []
    for batch_idx, (inputs, labels, time_feats, node_ids, time_starts) in enumerate(tqdm(testloader)):
        inputs = jnp.array(inputs.numpy())
        labels = jnp.array(labels.numpy())  # Not the most efficient...

        logits = model.apply({"params": state['params']}, inputs[:, :-1])
        # (bsz, seq_len - 1, 6)
        # breakpoint()
        loss = jnp.mean(cross_entropy_loss(logits.reshape(
            logits.shape[0], logits.shape[1], 6, 10), labels[:, 1:, :]))
        acc = jnp.mean(compute_accuracy(logits.reshape(logits.shape[0], logits.shape[1], 6, 10), labels[:, 1:, :]))

        losses.append(loss)
        accuracies.append(acc)

    loss_avg, acc_avg = jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies))
    print(f'validation: average loss {loss_avg:.5f}, average acc {acc_avg:.5f}')
