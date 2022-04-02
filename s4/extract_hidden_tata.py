"""Extract hidden states on data using trained models."""
import argparse
from itertools import chain
import os
from functools import partial
from typing import Dict

import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from tqdm import tqdm

from .s4 import BatchSeqModelHidden, S4LayerInit
from .data_tata import create_tata_dataset


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

# checkpoint_name = ('s4-d_model=128-sl512-dp0.05-bsz256/'
#                    'best_94')
# seq_length = 512
# bsz = 256
# d_model = 128


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


def get_feat_hidden_pretrained(model, state, xatu_features_nodes: Dict):
    """Get the hidden states (last layer before softmax) from the pretrained model,
    for every time step."""
    out_all_hidden_dict = {}

    for node_id, data_dict in tqdm(xatu_features_nodes.items()):
        data = jnp.array(data_dict['data_for_model'].numpy())

        last_hidden = model.apply({'params': state['params']}, data[:, 1:])    # (bsz, seq_length - 1, d_model)

        out_all_hidden_dict[node_id] = torch.tensor(np.asarray(last_hidden))

        # breakpoint()

    return out_all_hidden_dict


def make_xatu_model_hidden_states_stack(xatu_features_nodes: Dict, out_all_hidden_dict: Dict):
    """Stack hidden states features predicted by the model for Xatu for all nodes. This is used to train a simple
    log-linear model.

    Args:
        xatu_features_nodes (Dict): [description]
        out_all_hidden_dict (Dict): [description]
    """
    node_id_list = sorted(list(xatu_features_nodes.keys()))
    index_in_xatu = list(chain.from_iterable(
        [xatu_features_nodes[node_id]['index_in_xatu'] for node_id in node_id_list]))
    labels_in_xatu = list(chain.from_iterable(
        [xatu_features_nodes[node_id]['labels_in_xatu'] for node_id in node_id_list]))
    feats_stdz_tfc = torch.cat([xatu_features_nodes[node_id]['data_original_stdz'] for node_id in node_id_list], dim=0)

    feats_hidden = torch.cat([out_all_hidden_dict[node_id] for node_id in node_id_list], dim=0)    # (bsz, 512, 6)

    xatu_tfc_feats_all = {
        'index_in_xatu': index_in_xatu,
        'labels': labels_in_xatu,
        'feats_real': feats_stdz_tfc,    # focusing from the 1st step on
        'feats_hidden': feats_hidden,
    }
    return xatu_tfc_feats_all


if __name__ == '__main__':
    args = parse_args()

    # ========== where to save
    save_dir = f'/n/tata_ddos_ceph/tata_data/dataset/2019050607_v1_unsup_xatu_s4-d{args.d_model}-sl{args.seq_length}-out'
    os.makedirs(save_dir, exist_ok=True)

    # ========== Load data

    # # Create dataset...
    # trainloader, testloader, n_classes, seq_len, in_dim = create_tata_dataset(
    #     seq_length=args.seq_length,
    #     bsz=args.bsz,
    # )

    # ========== load the model

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)

    model = S4LayerInit(N=64)
    model = partial(
        BatchSeqModelHidden,
        layer=model,
        d_output=10 * 6,    # not used
        d_model=args.d_model,
        n_layers=args.n_layers,
        l_max=args.seq_length - 1,
    )
    rng = jax.random.PRNGKey(0)
    state = checkpoints.restore_checkpoint(checkpoint_path, None)
    model = model(training=False)

    # breakpoint()

    # # ========== apply model to data
    # losses, accuracies = [], []
    # for batch_idx, (inputs, labels, time_feats, node_ids, time_starts) in enumerate(tqdm(testloader)):
    #     inputs = jnp.array(inputs.numpy())
    #     labels = jnp.array(labels.numpy())  # Not the most efficient...

    #     hidden = model.apply({"params": state['params']}, inputs)
    #     # (bsz, seq_len, d_model)
    #     breakpoint()

    # ========== load prepared Xatu data
    xatu_batch_dir = '/n/tata_ddos_ceph/tata_data/dataset/2019050607_v1_unsup_xatu'
    seq_len = args.seq_length

    xatu_batch_val_path = os.path.join(xatu_batch_dir, f'xatu_val_feats_sl{seq_len}_nodes.pt')
    xatu_batch_test_path = os.path.join(xatu_batch_dir, f'xatu_test_feats_sl{seq_len}_nodes.pt')

    print(f'loading Xatu val and test node batch data from {xatu_batch_dir} ---')
    xatu_features_nodes_val = torch.load(xatu_batch_val_path)
    xatu_features_nodes_test = torch.load(xatu_batch_test_path)

    # ========== run the S4 model to get hidden states from the final layer
    print('Running pretrained S4 model to get last hidden states for Xatu val data ---')
    out_all_hidden_dict_val = get_feat_hidden_pretrained(model, state, xatu_features_nodes_val)
    save_path = os.path.join(save_dir, 'xatu_val_model_all_hidden.pt')
    torch.save(out_all_hidden_dict_val, save_path)
    print('-' * 10 + '> ' + f'saved at {save_path}')

    # breakpoint()

    print('Running pretrained model to get last hidden states for Xatu test data ---')
    out_all_hidden_dict_test = get_feat_hidden_pretrained(model, state, xatu_features_nodes_test)
    save_path = os.path.join(save_dir, 'xatu_test_model_all_hidden.pt')
    torch.save(out_all_hidden_dict_test, save_path)
    print('-' * 10 + '> ' + f'saved at {save_path}')

    # ========== make features including the model predictions for training a downstream simple classifier
    xatu_tfc_feats_all_val = make_xatu_model_hidden_states_stack(
        xatu_features_nodes_val, out_all_hidden_dict_val)

    save_path = os.path.join(save_dir, 'xatu_val_all_model_hidden_feats_and_labels.pt')
    torch.save(xatu_tfc_feats_all_val, save_path)
    print(f'extracted xatu val S4 last layer hidden states features all stacked by nodes saved at {save_path}')

    xatu_tfc_feats_all_test = make_xatu_model_hidden_states_stack(
        xatu_features_nodes_test, out_all_hidden_dict_test)

    save_path = os.path.join(save_dir, 'xatu_test_all_model_hidden_feats_and_labels.pt')
    torch.save(xatu_tfc_feats_all_test, save_path)
    print(f'extracted xatu test S4 last layer hidden states features all stacked by nodes saved at {save_path}')
