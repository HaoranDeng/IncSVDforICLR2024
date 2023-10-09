"""Script to run experiments for updating the truncated SVD of evolving matrices.
"""

import os
from os import mkdir
from os.path import normpath, join
import json
import numpy as np
from tqdm import tqdm
import scipy.sparse

import truncatedSVD.EvolvingMatrix as EM
from truncatedSVD.plotter import *
from truncatedSVD.utils import check_and_create_dir, p_r_curve, save_result
from string import Template
from assert_tests import validate_experiment

from scipy import spatial

np.random.seed(0)

def perform_updates(
    dataset,
    n_batches,
    model,
    method,
    update_method,
    network=False,
    **kwargs
):
    """Perform updates for specified number of batches using update method."""
    pbar = tqdm(range(n_batches))
    for ii in pbar:
        # Evolve matrix by appending new rows
        if network:
            model.evolve_network()
            model.update_matrix = model.E1
            update_method()
            model.Uk, model.Vk = model.Vk, model.Uk
            if model.Ku is not None:
                model.Ku, model.Kv = model.Kv, model.Ku

            model.update_matrix = model.E2
            update_method()
            model.Uk, model.Vk = model.Vk, model.Uk
            if model.Ku is not None:
                model.Ku, model.Kv = model.Kv, model.Ku

        else:
            model.evolve()
            update_method()

        pbar.set_postfix(runtime=model.runtime, length=f"{model.n_appended_total}/{model.s_dim}",restart = model.num_restart, svd_time = model.svd_time)
    print(f"Runtime: {model.runtime}")


def split_data(A, m_percent):
    """Split data row-wise"""
    # Calculate index of split
    m_dim_full = np.shape(A)[0]
    m_dim = int(np.ceil(m_dim_full * m_percent))

    # Split into initial matrix and matrix to be appended
    B = A[:m_dim, :]
    E = A[m_dim:, :]

    return B, E


def print_message(dataset, data_shape, method, n_batches, k, r=None):
    """Print details for current experiment."""
    print(100 * "*")
    print("")
    print(f"Dataset:           {dataset} {data_shape}")
    print(f"Update method:     {method}")
    print(f"Number of batches: {n_batches}")
    print(f"Rank k of updates: {k}")
    if r is not None:
        print(f"r (BCG parameter): {r}")
    print()


def run_experiments(specs_json, cache_path):
    test_spec = validate_experiment(specs_json)

    # Create cache path to save results
    cache_dir = join(cache_path, "cache")
    check_and_create_dir(cache_dir)

    # Loop through each dataset
    for test in test_spec["tests"]:
        dataset = test["dataset"]
        check_and_create_dir(join(cache_dir, dataset))

    
        if test["network"]:
            def load_adjacency_matrix(file):
                import h5py
                M = h5py.File(file, "r")['A']
                data, ir, jc = M['data'], M['ir'], M['jc']
                M = scipy.sparse.csc_matrix((data, ir, jc))
                return M
            data = load_adjacency_matrix(f"datasets/graph/{dataset}/train.mat")
            print("Network")
        else:
            # Load data
            filename = test_spec["dataset_info"][dataset + "_text"]
            data = scipy.sparse.load_npz(filename)
            print("*" * 100)
            print("")
            print(f"Filename: {filename}")
            print("")
            # data=scipy.sparse.rand(10000, 10000, density=0.0001,format="coo", dtype=None)
            data = data.T.tocsr()
        runtime_list = []

        # Run tests for each update method
        for method in test["methods"]:
            check_and_create_dir(join(cache_dir, dataset, method))

            # Loop through number of batches
            for n_batches in test["n_batches"]:
                # Calculate data split index
                if test["network"] == False:
                    B, E = split_data(data, test["m_percent"])

                # Loop through desired rank k
                for k in test["k_dims"]:
                   # Create directory to save data for this batch split and k
                    results_dir = join(
                        cache_dir,
                        dataset,
                        method,
                        f"{dataset}_n_batches_{str(n_batches)}_k_dims_{str(k)}",
                    )
                    check_and_create_dir(results_dir)
                    
                    # Update truncated SVD using Frequent Directions
                    if method == "isvd1":
                        print_message(dataset, data.shape, method, n_batches, k)
                        if test["network"] == False:
                            model = EM.EvolvingMatrix(B, n_batches=n_batches, k_dim=k, name="isvd1", max_rows=data.shape[0])
                            model.set_append_matrix(E)
                        else:
                            init = int(data.shape[0] * test["m_percent"])
                            model = EM.EvolvingMatrix(data[:init, :init], n_batches=n_batches, k_dim=k, name="isvd1", max_rows=data.shape[0], network=True)
                            A_csr = data.tocsr()
                            A_csc = data.tocsc()
                            model.set_network_append_matrix(A_csr=A_csr, A_csc=A_csc, init=init)

                        print()

                        perform_updates(
                            dataset,
                            n_batches,
                            model,
                            method,
                            model.update_svd_isvd1,
                            network=test["network"],
                        )

                    elif method == "isvd2":
                        print_message(dataset, data.shape, method, n_batches, k)
                        if test["network"] == False:
                            model = EM.EvolvingMatrix(B, n_batches=n_batches, k_dim=k, name="isvd2", max_rows=data.shape[0])
                            model.set_append_matrix(E)
                        else:
                            init = int(data.shape[0] * test["m_percent"])
                            model = EM.EvolvingMatrix(data[:init, :init], n_batches=n_batches, k_dim=k, name="isvd2", max_rows=data.shape[0], network=True)
                            A_csr = data.tocsr()
                            A_csc = data.tocsc()
                            model.set_network_append_matrix(A_csr=A_csr, A_csc=A_csc, init=init)

                        print()

                        perform_updates(
                            dataset,
                            n_batches,
                            model,
                            method,
                            model.update_svd_isvd2,
                            network=test["network"],
                        )

                    
                    elif method == "isvd3":
                        print_message(dataset, data.shape, method, n_batches, k)
                        if test["network"] == False:
                            model = EM.EvolvingMatrix(B, n_batches=n_batches, k_dim=k, name="isvd3", max_rows=data.shape[0])
                            model.set_append_matrix(E)
                        else:
                            init = int(data.shape[0] * test["m_percent"])
                            model = EM.EvolvingMatrix(data[:init, :init], n_batches=n_batches, k_dim=k, name="isvd3", max_rows=data.shape[0], network=True)
                            A_csr = data.tocsr()
                            A_csc = data.tocsc()
                            model.set_network_append_matrix(A_csr=A_csr, A_csc=A_csc, init=init)

                        print()

                        perform_updates(
                            dataset,
                            n_batches,
                            model,
                            method,
                            model.update_svd_isvd3,
                            network=test["network"],
                        )



                    # Update truncated SVD using Kalatanzis' Algorithm-1 variation 
                    elif method == "zhasimon":
                        print_message(dataset, data.shape, method, n_batches, k)
                        if test["network"] == False:
                            model = EM.EvolvingMatrix(B, n_batches=n_batches, k_dim=k, name="zha-simon", max_rows=data.shape[0])
                            model.set_append_matrix(E)
                        else:
                            init = int(data.shape[0] * test["m_percent"])
                            model = EM.EvolvingMatrix(data[:init, :init], n_batches=n_batches, k_dim=k, name="zha-simon", max_rows=data.shape[0], network=True)
                            A_csr = data.tocsr()
                            A_csc = data.tocsc()
                            model.set_network_append_matrix(A_csr=A_csr, A_csc=A_csc, init=init)

                        print()

                        perform_updates(
                            dataset,
                            n_batches,
                            model,
                            method,
                            model.update_svd_zhasimon,
                            network=test["network"],
                        )

                    elif method == "random":
                        print_message(dataset, data.shape, method, n_batches, k)
                        if test["network"] == False:
                            model = EM.EvolvingMatrix(B, n_batches=n_batches, k_dim=k, name="random", max_rows=data.shape[0])
                            model.set_append_matrix(E)
                        else:
                            init = int(data.shape[0] * test["m_percent"])
                            model = EM.EvolvingMatrix(data[:init, :init], n_batches=n_batches, k_dim=k, name="random", max_rows=data.shape[0], network=True)
                            A_csr = data.tocsr()
                            A_csc = data.tocsc()
                            model.set_network_append_matrix(A_csr=A_csr, A_csc=A_csc, init=init)

                        print()

                        perform_updates(
                            dataset,
                            n_batches,
                            model,
                            method,
                            model.update_random,
                            network=test["network"],
                        )

                    elif method == "vecharynski":
                        print_message(dataset, data.shape, method, n_batches, k)
                        if test["network"] == False:
                            model = EM.EvolvingMatrix(B, n_batches=n_batches, k_dim=k, name="vecharynski", max_rows=data.shape[0])
                            model.set_append_matrix(E)
                        else:
                            init = int(data.shape[0] * test["m_percent"])
                            model = EM.EvolvingMatrix(data[:init, :init], n_batches=n_batches, k_dim=k, name="vecharynski", max_rows=data.shape[0], network=True)
                            A_csr = data.tocsr()
                            A_csc = data.tocsc()
                            model.set_network_append_matrix(A_csr=A_csr, A_csc=A_csc, init=init)

                        print()

                        perform_updates(
                            dataset,
                            n_batches,
                            model,
                            method,
                            model.update_vecharynski,
                            network=test["network"],
                        )
                            
                    # Update method specified does not exist
                    else:
                        raise ValueError(
                            f"Update method {method} does not exist. "
                        )

                    if test["save_result"]:
                        save_dir_name = f"{cache_dir}/{dataset}/{method}/b{n_batches}_k{k}"
                        check_and_create_dir(save_dir_name)
                        
                        np.save(os.path.join(save_dir_name, "U.npy"), model.Uk)
                        np.save(os.path.join(save_dir_name, "V.npy"), model.Vk)
                        np.save(os.path.join(save_dir_name, "S.npy"), model.sigmak)
                        if model.Ku is not None:
                            np.save(os.path.join(save_dir_name, "ku.npy"), model.Ku)
                            np.save(os.path.join(save_dir_name, "kv.npy"), model.Kv)
                            
                        print(f"Save result to: {save_dir_name}")
                    runtime_list.append(model.runtime)
                    print(f"SVD time: {model.svd_time}")
        print(runtime_list)

######################################################################

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Run experiments for updating truncated SVD of evolving matrices.")

    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="specs_json",
        required=True,
        help="Experiment specifications. This file specifies all configurations for the experiments to run."
    )
    arg_parser.add_argument(
        "--cache_dir",
        "-c",
        dest="cache_dir",
        default=".",
        help="Directory to contain cache folder. A folder named 'cache' will be created to save all results."
    )

    args = arg_parser.parse_args()

    run_experiments(args.specs_json, args.cache_dir)
    print("Done.")
