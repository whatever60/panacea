import torch
from torch import nn
from torch import distributed as dist


class CSyncBatchNorm(nn.SyncBatchNorm):
    """Center batch norm.
    In each forward pass during training, norm is calculated as during eval,
    but statistics (mean and std) are updated normally.

    Set `with_var = False` to disable scaling.
    """

    def __init__(self, *args, with_var=False, **kwargs):
        super(CSyncBatchNorm, self).__init__(*args, **kwargs)
        self.with_var = with_var

    def forward(self, x):
        # center norm (instead of batch norm)
        self.training = False
        if not self.with_var:  # ignore variance
            self.running_var = torch.ones_like(self.running_var)
        normed_x = super(CSyncBatchNorm, self).forward(x)
        # udpate center
        self.training = True
        _ = super(CSyncBatchNorm, self).forward(x)
        return normed_x


class PSyncBatchNorm(nn.SyncBatchNorm):
    """
    Partial sync batch norm.
    BatchNorm won't by syncronized globally. A compromise between SyncBatchNorm and vanilla BatchNorm.

    ```
    bunch_size = 4
    world_size = 64
    current_rank = 42
    procs_per_bunck = 4
    n_bunch = 16
    rank_groups = [[0, 1, 2, 3], [4, 5, 6, 7], ..., [60, 61, 62, 63]]
    bunch_id = 10
    process_groups = [group0, ... group15]
    process_group = group[9]
    ```
    """

    def __init__(self, *args, bunch_size, **kwargs):
        procs_per_bunch = min(bunch_size, self.get_world_size())
        assert self.get_world_size() % procs_per_bunch == 0
        n_bunch = self.get_world_size() // procs_per_bunch
        #
        ranks = list(range(self.get_world_size()))
        print("---ALL RANKS----\n{}".format(ranks))
        rank_groups = [
            ranks[i * procs_per_bunch : (i + 1) * procs_per_bunch]
            for i in range(n_bunch)
        ]
        print("---RANK GROUPS----\n{}".format(rank_groups))
        process_groups = [torch.distributed.new_group(pids) for pids in rank_groups]
        bunch_id = self.get_rank() // procs_per_bunch
        process_group = process_groups[bunch_id]
        print("---CURRENT GROUP----\n{}".format(process_group))
        super(PSyncBatchNorm, self).__init__(
            *args, process_group=process_group, **kwargs
        )

    @staticmethod
    def is_dist_avail_and_initialized():
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    @classmethod
    def get_world_size(cls):
        if not cls.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    @classmethod
    def get_rank(cls):
        if not cls.is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()


class CustomSequential(nn.Sequential):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        # input: [batch_size, *whatever, C]
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                # ======================
                # This piece of code permutes the last dimension (channel) to the second
                # one, applies batchnorm on shape[2:], and permute backs.
                perm = list(range(dim - 1))
                perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]
                inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
                # ======================
            else:
                input = module(input)
        return input
