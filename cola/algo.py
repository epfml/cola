import warnings
import numpy as np

from . import communication as comm


def run_algorithm(algorithm, Ak, b, solver, gamma, theta, max_global_steps, local_iters, n_nodes, graph, monitor):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            'Objective did not converge. You might want to increase the number of iterations. '
            'Fitting data with very small alpha may cause precision problems.')

        comm.barrier()
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
    rank = comm.get_rank()

    # Initialize
    xk = np.zeros(n_cols)
    Akxk = np.zeros(n_rows)

    # Keep a list of neighborhood and their estimates of v
    local_lookups = graph.get_neighborhood(rank)
    local_vs = {node_id: np.zeros(n_rows) for node_id, _ in local_lookups.items()}

    sigma = gamma * K
    localsolver.dist_init(Ak, b, theta, local_iters, sigma)

    # Initial
    comm.p2p_communicate_neighborhood_tensors(
        rank, local_lookups, local_vs)

    monitor.log(np.zeros(n_rows), Akxk, xk, 0, localsolver)
    for i_iter in range(1, 1 + global_iters):
        # Average the local estimates of neighborhood and self
        averaged_v = comm.local_average(n_rows, local_lookups, local_vs)

        # Solve the suproblem using this estimates
        delta_x, delta_v = localsolver.solve(averaged_v, Akxk, xk)

        # update local variables
        xk += gamma * delta_x
        Akxk += gamma * delta_v

        # update shared variables
        averaged_v += gamma * delta_v * K
        local_vs[rank] = averaged_v
        comm.p2p_communicate_neighborhood_tensors(
            rank, local_lookups, local_vs)

        if monitor.log(averaged_v, Akxk, xk, i_iter, localsolver):
            print('break iterations here.')
            break

        if (i_iter % monitor.ckpt_freq) == 0:
            monitor.save(Akxk, xk, weightname='weight_epoch_{}.npy'.format(i_iter))

    return Akxk, xk
