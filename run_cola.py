import click
import numpy as np
import pandas as pd
import cola.communication as comm

from cola.dataset import load_dataset, load_dataset_by_rank
from cola.graph import define_graph_topology
from cola.cocoasolvers import configure_solver
from cola.algo import run_algorithm
from cola.monitor import Monitor


@click.command()
@click.option('--dataset', type=click.STRING, help='The type of dataset.')
@click.option('--solvername', type=click.STRING, help='The name of solvers.')
@click.option('--algoritmname', type=click.STRING, help='The name of algorithm')
@click.option('--output_dir', type=click.STRING, default=None, help='Save metrics in the training.')
@click.option('--dataset_size', default='small', type=click.Choice(['small', 'all']), help='Size of dataset')
@click.option('--use_split_dataset', is_flag=True)
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
@click.option('--n_connectivity', type=int, help='Connected Cycle.')
@click.option('--l1_ratio', type=float, help='l1 ratio in the ElasticNet')
@click.option('--lambda_', type=float, help='Size of regularizer')
@click.option('--c', type=float, help='Constant in the LinearSVM.')
@click.option('--ckpt_freq', type=int, default=10, help='')
@click.option('--exit_time', default=1000.0, help='The maximum running time of a node.')
def main(dataset, dataset_path, dataset_size, use_split_dataset, split_by, random_state,
         algoritmname, max_global_steps, local_iters, solvername, output_dir, exit_time, lambda_, l1_ratio, theta,
         graph_topology, c, logmode, ckpt_freq, n_connectivity):

    # Fix gamma = 1.0 according to:
    #   Adding vs. Averaging in Distributed Primal-Dual Optimization
    gamma = 1.0

    # Initialize process group
    comm.init_process_group('mpi')

    # Get rank of current process
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    # Create graph with specified topology
    graph = define_graph_topology(world_size, graph_topology, n_connectivity=n_connectivity)

    if use_split_dataset:
        X, y = load_dataset_by_rank(dataset, rank, world_size, dataset_size, split_by,
                                    dataset_path=dataset_path, random_state=random_state)
    else:
        X, y = load_dataset(dataset, rank, world_size, dataset_size, split_by,
                            dataset_path=dataset_path, random_state=random_state)

    # Define subproblem
    solver = configure_solver(name=solvername, split_by=split_by, l1_ratio=l1_ratio,
                              lambda_=lambda_, C=c, random_state=random_state)

    # Add hooks to log and save metrics.
    monitor = Monitor(solver, output_dir, ckpt_freq, exit_time, split_by, logmode)

    # Always use this value throughout this project
    Akxk, xk = run_algorithm(algoritmname, X, y, solver, gamma, theta,
                             max_global_steps, local_iters, world_size,
                             graph, monitor)

    monitor.save(Akxk, xk, weightname='weight.npy', logname='result.csv')

    # ------------------------------------------------------------
    # TODO:
    # def compute_l2_logistic_loss(X, y, w, C):
    #     pred = y * (X @ w)
    #     output = C * np.log(np.exp(- pred) + 1).sum()
    #     output += 0.5 * np.linalg.norm(w, 2) ** 2
    #     return output

    # print('Akxk.shape', Akxk.shape)
    # print('X.T.shape', X.T.shape)
    # print('Akxk[:10]', Akxk[:10])
    # print('c', c)
    # print('X[:5, :5]', X[:5, :5])
    # print('compute_svm_loss', compute_svm_loss(X.T, y, Akxk, c) * 1e-3)
    # c = 1 / (lambda_ * len(y))
    # print('compute_l2_logistic_loss', compute_l2_logistic_loss(X, y, xk, C=c) * lambda_)


if __name__ == '__main__':
    main()
