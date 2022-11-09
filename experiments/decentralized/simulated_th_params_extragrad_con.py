import argparse
import numpy as np
import pickle

from exp_params import N_ONE, DIM, MAT_MEAN, MAT_STD, EPS
from experiment_utils import run_extragrad_con, metropolis_weights, grid_adj_mat, star_adj_mat, \
    ring_adj_mat


def exp_extragrad_con(graph: str, num_nodes: int, noise: float):
    folder = "./logs/{}_nodes={}_noise={:.2e}".format(graph, num_nodes, noise)
    with open("{}/z_true".format(folder), "rb") as f:
        z_true = pickle.load(f)

    if graph == "star":
        mix_mat = metropolis_weights(star_adj_mat(num_nodes))
    elif graph == "grid":
        width = int(np.sqrt(num_nodes))
        mix_mat = metropolis_weights(grid_adj_mat(width, width))
    elif graph == "ring":
        mix_mat = metropolis_weights(ring_adj_mat(num_nodes))
    else:
        raise ValueError("Unknown graph type '{}'".format(graph))

    runner = run_extragrad_con(
        n_one=N_ONE,
        d=DIM,
        mat_mean=MAT_MEAN,
        mat_std=MAT_STD,
        noise=noise,
        num_nodes=num_nodes,
        mix_mat=mix_mat,
        regcoef_x=2.,
        regcoef_y=2.,
        r_x=5.,
        r_y=5.,
        eps=EPS,
        comm_budget_experiment=200000,
        z_true=z_true
    )

    with open("/".join((folder, "extragrad_con_th.pkl")), "wb") as f:
        pickle.dump(runner.logger, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running decentralized extragradient-con")

    parser.add_argument("--graph", type=str, nargs='?', dest="graph",
                        help="Graph type", required=True)
    args = parser.parse_args()

    exp_extragrad_con(args.graph, 25, 0.0001)
    exp_extragrad_con(args.graph, 25, 0.001)
    exp_extragrad_con(args.graph, 25, 0.01)
    exp_extragrad_con(args.graph, 25, 0.1)
    exp_extragrad_con(args.graph, 25, 1.)
    exp_extragrad_con(args.graph, 25, 10.)
