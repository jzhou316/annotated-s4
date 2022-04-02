import os
from xml.dom import NotFoundErr
from itertools import chain
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

import torch
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, BatchSampler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# total number of nodes, for each time segment
NUM_NODES = 1351


# ### Tata netflow traffic sequence modeling
# **Task**: Predict next time step (min) netflow traffic for each IP node (traffic time series modeling).
#
# Input features have 11 dimensions (6 traffic features, 4 time features, 1 node id). Output 6 traffic feature
# predictions, each of 10 discrete levels.
def create_tata_dataset(seq_length=512, bsz=128):
    print("[*] Generating Tata netflow Sequence Modeling Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = seq_length, 10 * 6, 11
    TIME_IDX_END, TIME_IDX_VAL_BEG = 100337, 90112
    # train time period: [0, TIME_IDX_VAL_BEG), val time period: [TIME_IDX_VAL_BEG, TIME_IDX_END) (ratio ~ 8.8 : 1)

    data_dir = 'data/tata_netflow/2019050607'
    if (not os.path.exists(os.path.join(data_dir, 'tfc-deg_aggr-nodes_sub1-io_disc-10.pt'))
        or not os.path.exists(os.path.join(data_dir, 'time_features_discrete.pt'))
        ):
        raise NotFoundErr(f'Tata netflow dataset not found in {data_dir}')

    traffic_feats = torch.load(os.path.join(data_dir, 'tfc-deg_aggr-nodes_sub1-io_disc-10.pt'))
    time_feats = torch.load(os.path.join(data_dir, 'time_features_discrete.pt'))

    num_nodes, num_times, num_traffic_feats = traffic_feats.size()    # 1351, 128207, 6
    num_times, num_time_feats = time_feats.size()    # 128207, 4

    # cut the original time period for training and validation (to be consistent with the pre-training setup)
    traffic_feats = traffic_feats[:, :TIME_IDX_END, :]
    time_feats = time_feats[:TIME_IDX_END, :]

    train_data = []
    valid_data = []

    # break data into equal time chunks (last chunk might have shorter time)
    print(' ' * 4 + f'- breaking time series into equal chunks of lengths {SEQ_LENGTH}')
    stride = SEQ_LENGTH    # non-overlapping time chunks
    start = 0
    in_train = True
    num_segments_train = 0
    num_segments_valid = 0
    while start < TIME_IDX_END:
        if start >= TIME_IDX_VAL_BEG:
            in_train = False

        end = start + SEQ_LENGTH

        if in_train and end > TIME_IDX_VAL_BEG:
            # some time steps entering validation period
            end = TIME_IDX_VAL_BEG

        # formulate data example: breakdown nodes as well
        traffic_chunk_nodes = torch.chunk(traffic_feats[:, start:end, :], num_nodes, dim=0)    # length 1351 list
        time_chunk = time_feats[start:end, :]    # a single tensor

        if in_train:
            num_segments_train += 1
        else:
            num_segments_valid += 1

        # append to dataset
        for node_id, traffic_chunk in enumerate(traffic_chunk_nodes):
            data_example = (traffic_chunk, time_chunk, node_id, start)
            # traffic_chunk: size (1, seq_len, 6)
            # time_chunk: size (seq_len, 6)
            # node_id: int
            # start: int

            # stack features to formulate batch example (default collate_fn will prepend a new batch dim)
            seq_len = time_chunk.size(0)
            data_item = torch.cat([traffic_chunk.squeeze(0), time_chunk, torch.zeros(seq_len, 1) + node_id], dim=1)
            data_item_example = (data_item, traffic_chunk.squeeze(0), time_chunk, node_id, start)

            if in_train:
                train_data.append(data_item_example)
            else:
                valid_data.append(data_item_example)

        start += stride

    print(' ' * 4 + f'- number of time segments for training: {num_segments_train} '
          f'(last segment length: {train_data[-1][0].size(0)})')
    print(' ' * 4 + f'- number of time segments for validation: {num_segments_valid} '
          f'(last segment length: {valid_data[-1][0].size(0)})')
    print(' ' * 4 + f'- total number of time series for training: {num_segments_train * num_nodes} ({num_nodes} nodes)')
    print(' ' * 4 + f'- total number of time series for validation: {num_segments_valid * num_nodes} ({num_nodes} nodes)')

    # breakpoint()

    # create dataset
    # train_data
    # valid_data

    # create data loader
    # NOTE in jax, every batch should be of the same size, so we have to drop the last bucket of batch
    trainloader = TwoBucketDataLoader(
        train_data, batch_size=bsz, shuffle=True, drop_last=False, drop_last_bucket=False
    )
    testloader = TwoBucketDataLoader(
        valid_data, batch_size=bsz, shuffle=False, drop_last=False, drop_last_bucket=True
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### handing the data batch iteration
# the time series might have different lengths (due to that we chunk a long time series, the last chunk may not have
# the assigned sequence length), and we want to only put time series of the same length in a minibatch (we do not do
# padding in a mini-batch), while being able to randomly shuffling the data.

class TwoBucketRandomSampler(Sampler[int]):
    """Randomly shuffle the indices in two buckets separately, and then concatenate the two shuffled lists.

    Args:
        Sampler ([type]): [description]
    """

    def __init__(self,
                 data_source: Sized,
                 last_bucket_size: int = NUM_NODES,
                 shuffle: bool = True) -> None:
        assert len(data_source) % last_bucket_size == 0, \
            f'Each time segment for all nodes must have {last_bucket_size} pieces'
        self.data_source = data_source
        self.last_bucket_size = last_bucket_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            # equivalent to a SequentialSampler
            return iter(range(len(self.data_source)))
        else:
            # randomly shuffle the indices with each bucket
            first_bucket_len = len(self) - self.last_bucket_size
            first_bucket = torch.randperm(first_bucket_len).tolist()
            last_bucket = (torch.randperm(self.last_bucket_size) + first_bucket_len).tolist()
            return iter(first_bucket + last_bucket)

    def __len__(self) -> int:
        return len(self.data_source)


class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class TwoBucketBatchSampler(Sampler[List[int]]):
    """Yield mini-batch of indices, with two buckets, and a mini-batch could not contain cross-bucket indices.

    Args:
        Sampler ([type]): [description]
    """

    def __init__(self,
                 first_sampler: Sampler[int],
                 last_sampler: Sampler[int],
                 batch_size: int,
                 drop_last: bool,
                 drop_last_bucket: bool) -> None:

        self.first_batch_sampler = BatchSampler(first_sampler, batch_size=batch_size, drop_last=drop_last)
        self.last_batch_sampler = BatchSampler(last_sampler, batch_size=batch_size, drop_last=drop_last)

        self.drop_last_bucket = drop_last_bucket

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.first_batch_sampler:
            yield batch
        if not self.drop_last_bucket:
            for batch in self.last_batch_sampler:
                yield batch

        # if not self.drop_last_bucket:
        #     return chain(iter(self.first_batch_sampler), iter(self.last_batch_sampler))
        # else:
        #     return iter(self.first_batch_sampler)

    def __len__(self) -> int:
        if not self.drop_last_bucket:
            return len(self.first_batch_sampler) + len(self.last_batch_sampler)
        else:
            return len(self.first_batch_sampler)


class TwoBucketDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, drop_last_bucket=False, num_workers=0,
                 last_bucket_size=NUM_NODES):
        self.last_bucket_size = last_bucket_size
        self.first_bucket_size = len(dataset) - last_bucket_size

        # create two samplers for two buckets
        first_indices = list(range(self.first_bucket_size))
        last_indices = list(range(self.first_bucket_size, len(dataset)))

        if shuffle:
            first_sampler = SubsetRandomSampler(first_indices)
            last_sampler = SubsetRandomSampler(last_indices)
        else:
            first_sampler = SubsetSequentialSampler(first_indices)
            last_sampler = SubsetSequentialSampler(last_indices)

        # create batch sampler
        batch_sampler = TwoBucketBatchSampler(
            first_sampler, last_sampler, batch_size=batch_size, drop_last=drop_last, drop_last_bucket=drop_last_bucket)

        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            # collate_fn=collate_fn,
            num_workers=num_workers,
            )
