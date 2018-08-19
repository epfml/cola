import click
import numpy as np
import pandas as pd
import torch.distributed as dist

from cola.dataset import load_dataset
from cola.graph import define_graph_topology
from cola.cocoasolvers import configure_solver
from cola.algo import run_algorithm
from cola.monitor import Monitor


@click.command()
@click.option('--dataset', type=click.STRING, help='The type of dataset.')
@click.option('--solvername', type=click.STRING, help='The name of solvers.')
@click.option('--algoritmname', type=click.STRING, help='The name of algorithm')
@click.option('--logfile', type=click.STRING, default=None, help='Save metrics in the training.')
@click.option('--weightfile', type=click.STRING, default=None, help='Save weights on rank 0.')
@click.option('--dataset_size', default='small', type=click.Choice(['small', 'all']), help='Size of dataset')
@click.option('--logmode', default='local', type=click.Choice(['local', 'global']),
              help='Log local or global information.')
@click.option('--split_by', default='samples', type=click.Choice(['samples', 'features']),
              help='Split data matrix by samples or features.')
@click.option('--max_global_steps', default=100, help='Maximum number of global steps.')
@click.option('--theta', type=float, help='Theta-approximate solution (if local_iters is not specified)')
@click.option('--local_iters', default=1.0, help='Theta-approximate solution in terms of local data pass')
@click.option('--random_state', default=42, help='Random state')
@click.option('--dataset_path', default=None, help='Path to dataset')
@click.option('--graph_topology', type=str, help='Graph topology of the network.')
@click.option('--l1_ratio', type=float, help='l1 ratio in the ElasticNet')
@click.option('--lambda_', type=float, help='Size of regularizer')
@click.option('--c', type=float, help='Constant in the LinearSVM.')
@click.option('--exit_time', default=1000.0, help='The maximum running time of a node.')
def main(dataset, dataset_path, dataset_size, split_by, random_state,
         algoritmname, max_global_steps, local_iters, solvername, logfile, exit_time, lambda_, l1_ratio, theta,
         graph_topology, c, weightfile, logmode):

    # Fix gamma = 1.0 according to:
    #   Adding vs. Averaging in Distributed Primal-Dual Optimization
    gamma = 1.0

    # Initialize process group
    dist.init_process_group('mpi')

    # Get rank of current process
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create graph with specified topology
    graph = define_graph_topology(world_size, graph_topology)

    X, y = load_dataset(dataset, rank, world_size, dataset_size, split_by,
                        dataset_path=dataset_path, random_state=random_state)

    # Transpose the matrix depending on the matrix split direction
    if split_by == 'samples':
        X = X.T

    if not X.flags['F_CONTIGUOUS']:
        # The local coordinate solver (like scikit-learn's ElasticNet) requires X to be Fortran contiguous.
        # Since the time spent on converting the matrix can be very long, and CoCoA need to call solvers every round,
        # we perform such convertion before algorithm and disable check later.
        X = np.asfortranarray(X)

    # Define subproblem
    solver = configure_solver(name=solvername, split_by=split_by, l1_ratio=l1_ratio,
                              lambda_=lambda_, C=c, random_state=random_state)

    # Add hooks to log and save metrics.
    monitor = Monitor(solver, exit_time, split_by, logmode)

    # Always use this value throughout this project
    Akxk, xk = run_algorithm(algoritmname, X, y, solver, gamma, theta,
                             max_global_steps, local_iters, world_size, graph, monitor)

    monitor.save(logfile, weightfile, Akxk, xk)


if __name__ == '__main__':
    main()
