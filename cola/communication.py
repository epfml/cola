import numpy as np
# import torch
# import torch.distributed as dist

from mpi4py import MPI


def barrier():
    MPI.COMM_WORLD.Barrier()


def init_process_group(backend):
    pass


def get_rank():
    return MPI.COMM_WORLD.rank


def get_world_size():
    return MPI.COMM_WORLD.size


class CoCoAExitException(Exception):
    pass


def pytorch_local_average(n, local_lookup, local_tensors):
    """Average the neighborhood tensors.

    Parameters
    ----------
    n : {int}
        Size of tensor
    local_lookup : {dict: int->float}
        A dictionary from rank of neighborhood to the weight between two processes
    local_tensors : {dict: int->tensor}
        A dictionary from rank to tensors to be aggregated.

    Returns
    -------
    tensor
        An averaged tensor
    """
    averaged = torch.DoubleTensor(np.zeros(n))
    for node_id, node_weight in local_lookup.items():
        averaged += node_weight * local_tensors[node_id]
    return averaged


def local_average(n, local_lookup, local_tensors, backend=''):
    averaged = np.zeros(n)
    for node_id, node_weight in local_lookup.items():
        averaged += node_weight * local_tensors[node_id]
    return averaged


def p2p_communicate_neighborhood_tensors(rank, neighborhood, neighorhood_tensors, backend=''):
    """Communicate tensors with neighborhoods.

    Parameters
    ----------
    rank : {int}
        Rank of current process
    neighborhood : {dict: int->float}
        A dictionary from rank of neighborhood to the weight between two processes
    neighorhood_tensors : {dict: int->tensor}
        A dictionary from rank to tensors to be aggregated.
    """
    reqs = []
    for node_id in neighborhood:
        if node_id != rank:
            reqs.append(MPI.COMM_WORLD.Isend(neighorhood_tensors[rank], node_id))
            reqs.append(MPI.COMM_WORLD.Irecv(neighorhood_tensors[node_id], node_id))

    for req in reqs:
        req.Wait()


def pytorch_p2p_communicate_neighborhood_tensors(rank, neighborhood, neighorhood_tensors):
    """Communicate tensors with neighborhoods.

    Parameters
    ----------
    rank : {int}
        Rank of current process
    neighborhood : {dict: int->float}
        A dictionary from rank of neighborhood to the weight between two processes
    neighorhood_tensors : {dict: int->tensor}
        A dictionary from rank to tensors to be aggregated.
    """
    reqs = []
    for node_id in neighborhood:
        if node_id != rank:
            reqs.append(dist.isend(tensor=neighorhood_tensors[rank], dst=node_id))
            reqs.append(dist.irecv(tensor=neighorhood_tensors[node_id], src=node_id))

    for req in reqs:
        req.wait()


def all_reduce(data, op):
    if op == 'SUM':
        op = MPI.SUM
    elif op == 'MAX':
        op = MPI.MAX
    return MPI.COMM_WORLD.allreduce(data, op=op)


def reduce(data, op, root):
    if op == 'SUM':
        op = MPI.SUM
    elif op == 'MAX':
        op = MPI.MAX
    return MPI.COMM_WORLD.reduce(data, op=op, root=root)
