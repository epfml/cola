import warnings
import torch.distributed as dist
import numpy as np
import torch

from . import communication


def run_algorithm(algorithm, Ak, b, solver, gamma, theta, max_global_steps, local_iters, n_nodes, graph, monitor):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            'Objective did not converge. You might want to increase the number of iterations. '
            'Fitting data with very small alpha may cause precision problems.')

        dist.barrier()
        if algorithm == 'cola':
            Akxk, xk = cola(Ak, b, solver, gamma, theta, max_global_steps, local_iters, n_nodes, graph, monitor)
        elif algorithm == 'cocoa':
            Akxk, xk = cocoa(A_partition, b, solver, gamma, theta, max_global_steps, local_iters, n_nodes)
        elif algorithm == 'accelerated_cola':
            Akxk, xk = accelerated_cola(Ak, b, solver, gamma, theta, max_global_steps,
                                        local_iters, n_nodes, graph, monitor)
        else:
            raise NotImplementedError()
    return Akxk, xk


def cola(Ak, b, localsolver, gamma, theta, global_iters, local_iters, K, graph, monitor):
    if gamma <= 0 or gamma > 1:
        raise ValueError("gamma should in (0, 1]: got {}".format(gamma))

    # Shape of the matrix
    n_rows, n_cols = Ak.shape

    # Current rank of the node
    rank = dist.get_rank()

    # Initialize
    xk = torch.DoubleTensor(np.zeros(n_cols))
    Akxk = torch. DoubleTensor(np.zeros(n_rows))

    # Keep a list of neighborhood and their estimates of v
    local_lookups = graph.get_neighborhood(rank)
    local_vs = {node_id: torch.DoubleTensor(np.zeros(n_rows)) for node_id, _ in local_lookups.items()}

    sigma = gamma * K
    localsolver.dist_init(Ak, b, theta, local_iters, sigma)

    # Initial
    communication.pytorch_p2p_communicate_neighborhood_tensors(
        rank, local_lookups, local_vs)

    monitor.log(torch.DoubleTensor(n_rows), Akxk, xk, 0, localsolver)
    for i_iter in range(1, 1 + global_iters):
        # Average the local estimates of neighborhood and self
        averaged_v = communication.pytorch_local_average(n_rows, local_lookups, local_vs)

        # Solve the suproblem using this estimates
        delta_x, delta_v = localsolver.solve(averaged_v, Akxk, xk)
        delta_x, delta_v = torch.DoubleTensor(delta_x), torch.DoubleTensor(delta_v)

        # update local variables
        xk += gamma * delta_x
        Akxk += gamma * delta_v

        # update shared variables
        averaged_v += gamma * delta_v * K
        local_vs[rank] = averaged_v
        communication.pytorch_p2p_communicate_neighborhood_tensors(
            rank, local_lookups, local_vs)
        if monitor.log(averaged_v, Akxk, xk, i_iter, localsolver):
            break

        if (i_iter % monitor.ckpt_freq) == 0:
            monitor.save(Akxk, xk, weightname='weight_epoch_{}.npy'.format(i_iter))

    return Akxk, xk


def accelerated_cola(Ak, b, localsolver, gamma, theta, global_iters, local_iters, K, graph, monitor):
    if gamma <= 0 or gamma > 1:
        raise ValueError("gamma should in (0, 1]: got {}".format(gamma))

    # Shape of the matrix
    n_rows, n_cols = Ak.shape

    # Current rank of the node
    rank = dist.get_rank()

    # Initialize
    xk = torch.DoubleTensor(np.zeros(n_cols))
    Akxk = torch. DoubleTensor(np.zeros(n_rows))

    # Keep a list of neighborhood and their estimates of v
    local_lookups = graph.get_neighborhood(rank)
    local_vs = {node_id: torch.DoubleTensor(np.zeros(n_rows)) for node_id, _ in local_lookups.items()}

    sigma = gamma * K
    localsolver.dist_init(Ak, b, theta, local_iters, sigma)

    # Initial
    communication.pytorch_p2p_communicate_neighborhood_tensors(
        rank, local_lookups, local_vs)

    t_old = 1.0
    t_new = 1.0
    y_old = torch.DoubleTensor(np.zeros(n_cols))
    y_new = torch.DoubleTensor(np.zeros(n_cols))

    monitor.log(torch.DoubleTensor(n_rows), Akxk, xk, 0, localsolver)
    # Average the local estimates of neighborhood and self
    averaged_v = communication.pytorch_local_average(n_rows, local_lookups, local_vs)
    for i_iter in range(1, 1 + global_iters):

        # Solve the suproblem using this estimates
        delta_x, delta_v = localsolver.solve(averaged_v, Akxk, xk)

        delta_x, delta_v = torch.DoubleTensor(delta_x), torch.DoubleTensor(delta_v)

        y_new = xk + delta_x
        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2

        delta_x_ = delta_x + (t_old - 1) / t_new * (y_new - y_old)
        delta_v_ = torch.DoubleTensor(Ak @ delta_x_.numpy())

        # update local variables
        xk += gamma * delta_x_
        Akxk += gamma * delta_v_

        # update shared variables
        averaged_v += gamma * delta_v_ * K
        local_vs[rank] = averaged_v

        communication.pytorch_p2p_communicate_neighborhood_tensors(
            rank, local_lookups, local_vs)

        # Average the local estimates of neighborhood and self
        averaged_v = communication.pytorch_local_average(n_rows, local_lookups, local_vs)

        y_old = y_new
        t_old = t_new

        if monitor.log(averaged_v, Akxk, xk, i_iter, localsolver):
            break

        if (i_iter % monitor.ckpt_freq) == 0:
            monitor.save(Akxk, xk, weightname='weight_epoch_{}.npy'.format(i_iter))

    return Akxk, xk
