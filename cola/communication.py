import numpy as np
import torch
import torch.distributed as dist


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
