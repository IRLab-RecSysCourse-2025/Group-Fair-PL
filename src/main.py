# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import pyjson5
import datetime as dt
import json
import os
import pathlib
import time

import numpy as np
import torch
import torch.optim as optim

from .algorithms import PLRank_multiprocessing as plr

# import algorithms.tensorflowloss as tfl # Old TF loss
from .algorithms import pytorchloss as ptl  # New PyTorch loss
from .utils import dataset
from .utils import evaluate as evl  # Uses NumPy scores
from .utils import nnmodel as nn  # Now PyTorch model


def read_args():
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument(
        "--file",
        type=str,
        help="path to a config file for the algorithms",
        required=True,
    )
    parser.add_argument(
        "--loss",
        type=str.lower,
        help="name of the loss function",
        default="PL-Rank-3",
        choices=[
            "plrank3",
            "groupfairpl",
            "policygradient",
            "placementpolicygradient",
        ],
    )
    parser.add_argument(
        "-p",
        "--postprocess_algorithms",
        type=str,
        help="a comma-delimited list of postprocessing algorithms to apply. [options: none, GDL23, GAK19]",
        default="none,GDL23,GAK19",
    )
    parser.add_argument(
        "-b",
        "--bias",
        type=int,
        default=-1,
        help="bias component of (0.bias) the dataset; use --bias -1 for no bias",
    )
    parser.add_argument("-r", "--run_no", type=int, help="run number", default=1)
    parser.add_argument(
        "--device",
        type=torch.device,
        help="torch device to use for training [defaults to `cuda` if available]",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    # Read the JSONC config file
    with open(args.file, "r") as f:
        config = pyjson5.loads(f.read())

    if config["verbose"]:
        print(config)
    return args, config


def main():
    args, config = read_args()
    device = args.device

    num_samples = config["num_samples"]

    if num_samples == "dynamic":
        dynamic_samples = True
    else:
        dynamic_samples = False
        num_samples = int(num_samples)

    p = pathlib.Path("local_output/")
    p.mkdir(parents=True, exist_ok=True)

    if args.bias > -1:
        dataset_name = config["dataset"] + "biased_0_" + str(args.bias)
    else:
        dataset_name = config["dataset"]

    def read_dataset(config):
        print("Reading dataset...")
        data = dataset.get_dataset_from_json_info(
            dataset_name=dataset_name,
            info_path=config["dataset_info_path"],
            read_from_pickle=False,  # Set to True if you have pickles
        )
        fold_id = (config["fold_id"] - 1) % data.num_folds()
        data = data.get_data_folds()[fold_id]
        data.read_data()
        return data

    output_path = (
        "./local_output/"
        + dataset_name
        + "/"
        + str(dt.datetime.now())
        + "_loss="
        + args.loss
        + "_k="
        + str(config["cutoff"])
        + "_fairness="
        + config["fairness_requirement"]
        + "_nsamples="
        + str(config["num_samples"])
        + "_delta="
        + str(config["delta"])
        + "_run="
        + str(args.run_no)
        + ".json"
    )

    isExist = os.path.exists("./local_output/" + dataset_name + "/")
    if not isExist:
        os.makedirs("./local_output/" + dataset_name + "/")

    data = read_dataset(config)
    n_queries = data.train.num_queries()

    epoch_results = []
    timed_results = []

    max_ranking_size = np.min((config["cutoff"], data.max_query_size()))
    # metric_weights are NumPy, keep as is for now as evl and plr use NumPy
    metric_weights = (
        1.0 / np.log2(np.arange(max_ranking_size + 1) + 2)[:max_ranking_size]
    )
    train_labels = 2**data.train.label_vector - 1
    valid_labels = 2**data.validation.label_vector - 1
    test_labels = 2**data.test.label_vector - 1

    train_gender_labels = data.train.gender_vector
    valid_gender_labels = data.validation.gender_vector
    test_gender_labels = data.test.gender_vector
    n_groups = np.maximum(
        len(np.unique(train_gender_labels)), len(np.unique(valid_gender_labels))
    )
    if (
        n_groups == 0 and len(np.unique(train_gender_labels)) > 0
    ):  # handle case where validation might be empty of gender labels
        n_groups = len(np.unique(train_gender_labels))
    elif n_groups == 0 and len(np.unique(valid_gender_labels)) > 0:
        n_groups = len(np.unique(valid_gender_labels))
    elif (
        n_groups == 0
    ):  # if both are empty, default to 1 group to avoid division by zero
        n_groups = 1
        print(
            "Warning: No group labels found in train or validation. Defaulting to n_groups=1."
        )

    fairness_constraints = []
    prefix_fairness_constraints = []
    if n_groups > 0 and (len(train_gender_labels) + len(valid_gender_labels)) > 0:
        proportions = np.array([
            (
                len(np.where(train_gender_labels == j)[0])
                + len(np.where(valid_gender_labels == j)[0])
            )
            / (len(train_gender_labels) + len(valid_gender_labels))
            for j in range(n_groups)
        ])
    else:  # Default proportions if no gender labels or no items
        proportions = np.ones(n_groups) / n_groups if n_groups > 0 else np.array([1.0])

    if config["fairness_requirement"] == "Equal":
        for t in range(config["cutoff"]):
            lower = []
            upper = []
            for j in range(n_groups):
                l = t * (1 / n_groups - config["delta"])
                u = t * (1 / n_groups + config["delta"])
                lower.append(l)
                upper.append(u)
            prefix_fairness_constraints.append([lower, upper])
        fairness_constraints.append([
            int(np.floor(config["cutoff"] * (1 / n_groups - config["delta"])))
            for _ in range(n_groups)
        ])
        fairness_constraints.append([
            int(np.ceil(config["cutoff"] * (1 / n_groups + config["delta"])))
            for _ in range(n_groups)
        ])
    elif config["fairness_requirement"] == "Proportional":
        for t in range(config["cutoff"]):
            lower = []
            upper = []
            for j in range(n_groups):
                l = t * (proportions[j] - config["delta"])
                # u = t * (proportions[j] + config["delta"]) # Original
                u = l + 1  # As per original code for prefix_fairness_constraints
                lower.append(l)
                upper.append(
                    u
                )  # Corrected to use u, was using l+1 before, now using the calculated u
            prefix_fairness_constraints.append([lower, upper])
        fairness_constraints.append([
            int(np.floor(config["cutoff"] * (proportions[j] - config["delta"])))
            for j in range(n_groups)
        ])
        fairness_constraints.append([
            int(np.ceil(config["cutoff"] * (proportions[j] + config["delta"])))
            for j in range(n_groups)
        ])

    if (
        not fairness_constraints
    ):  # Ensure it's not empty if n_groups was 0 or 1 initially
        fairness_constraints = [[0] * n_groups, [config["cutoff"]] * n_groups]
    if not prefix_fairness_constraints:
        prefix_fairness_constraints = [[[0] * n_groups, [1] * n_groups]] * config[
            "cutoff"
        ]

    fairness_constraints = np.asarray(fairness_constraints, dtype=np.int32)
    prefix_fairness_constraints = np.asarray(
        prefix_fairness_constraints, dtype=np.int32
    )

    if config["verbose"]:
        print("Proportions:", proportions)
        print("Fairness Constraints:", fairness_constraints)
    ideal_train_metrics = evl.ideal_metrics(data.train, metric_weights, train_labels)
    ideal_valid_metrics = evl.ideal_metrics(
        data.validation, metric_weights, valid_labels
    )
    ideal_test_metrics = evl.ideal_metrics(data.test, metric_weights, test_labels)

    model_params = {
        "hidden units": [32, 32],
        "learning_rate": config["learning_rate"],
        "learning_rate_decay": 1.0,  # PyTorch scheduler will handle this differently
        "input_dim": data.train.feature_matrix.shape[1],  # Crucial for PyTorch
    }
    # model = nn.init_model(model_params) # Old TF model
    model = nn.init_torch_model(model_params).to(device)  # PyTorch model

    postprocess_algorithms = args.postprocess_algorithms.split(",")

    real_start_time = time.time()
    total_train_time = 0
    last_total_train_time = time.time()
    method_train_time = 0

    if num_samples == "dynamic":  # Original logic for dynamic samples
        dynamic_samples = True
        float_num_samples = 10.0
        # add_per_step = 90.0 / (n_queries * 40.0) # Original, might need adjustment
        add_per_step = 90.0 / (n_queries * 40.0) if n_queries > 0 else 0.01
        max_num_samples = 1000
    else:  # Ensure num_samples is int if not dynamic
        num_samples = int(num_samples)

    steps = 0
    batch_size = config["batch_size"]

    # PyTorch Optimizer
    optimizer = optim.SGD(model.parameters(), lr=model_params["learning_rate"])
    # PyTorch LR Scheduler (approximates TF's ExponentialDecay)
    # TF's decay_steps is per batch, PyTorch's ExponentialLR is per epoch.
    # For a closer match, you might need a LambdaLR or adjust gamma.
    # Here, we'll use a simple ExponentialLR per epoch.
    # decay_steps_tf = n_queries / batch_size
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=model_params["learning_rate_decay"]**(1.0/decay_steps_tf if decay_steps_tf > 0 else 1.0))
    # Simpler: decay gamma per epoch
    lr_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=model_params["learning_rate_decay"]
    )

    epoch_i = -1
    is_last_epoch = False
    while epoch_i < config["n_epochs"]:
        if config["verbose"]:
            print("EPOCH: %04d.00 TIME: %04d" % (epoch_i, total_train_time))
        epoch_i += 1

        model.train()  # Set model to training mode

        if dynamic_samples:  # Original dynamic sample logic
            # num_samples = int(max(1, np.ceil(200 * np.sqrt(epoch_i)))) # Original epoch-based
            # Use the step-based update primarily, re-evaluate if epoch-based is better
            pass  # num_samples will be updated per batch if dynamic

        query_permutation = np.random.permutation(n_queries)
        epoch_loss = 0.0
        num_batches = int(np.ceil(n_queries / batch_size))

        for batch_i in range(num_batches):
            optimizer.zero_grad()  # Zero gradients for each batch

            batch_queries = query_permutation[
                batch_i * batch_size : (batch_i + 1) * batch_size
            ]
            cur_batch_size = batch_queries.shape[0]
            if cur_batch_size == 0:
                continue

            batch_features_np = []
            for i in range(cur_batch_size):
                batch_features_np.append(data.train.query_feat(batch_queries[i]))

            batch_ranges = np.zeros(cur_batch_size + 1, dtype=np.int32)
            batch_ranges[1:] = [
                batch_features_np[idx].shape[0] for idx in range(cur_batch_size)
            ]
            batch_ranges = np.cumsum(batch_ranges)

            # Concatenate all document features into a single matrix (NumPy)
            batch_features_np_cat = np.concatenate(batch_features_np, axis=0)
            # Convert to PyTorch tensor
            batch_features_torch = torch.from_numpy(
                batch_features_np_cat.astype(np.float32)
            ).to(device)

            # Forward pass
            batch_torch_scores = model(
                batch_features_torch
            )  # Shape (total_docs_in_batch, 1)

            current_batch_loss = 0.0
            # For losses like PL-Rank that operate on NumPy arrays and are calculated per query
            batch_doc_weights_np = np.zeros(
                batch_features_np_cat.shape[0], dtype=np.float64
            )
            use_doc_weights_for_loss = (
                False  # Flag to indicate if loss is computed via doc_weights
            )

            for i, qid in enumerate(batch_queries):
                q_labels_np = data.train.query_values_from_vector(qid, train_labels)
                q_gender_labels_np = data.train.query_values_from_vector(
                    qid, train_gender_labels
                )
                # q_feat_np = batch_features_np_cat[batch_ranges[i] : batch_ranges[i + 1], :] # Already have this
                q_ideal_metric = ideal_train_metrics[qid]

                if q_ideal_metric != 0:
                    q_metric_weights_np = metric_weights  # NumPy

                    # Get scores for the current query from the batch_torch_scores
                    q_torch_scores = batch_torch_scores[
                        batch_ranges[i] : batch_ranges[i + 1]
                    ]

                    last_method_train_time = time.time()
                    if args.loss == "policygradient":
                        # ptl.policy_gradient expects PyTorch scores, NumPy labels/weights
                        loss_val = ptl.policy_gradient(
                            q_metric_weights_np,
                            q_labels_np,
                            q_torch_scores,  # PyTorch tensor
                            n_samples=num_samples,
                            device=device,
                        )
                        current_batch_loss += loss_val
                    elif args.loss == "placementpolicygradient":
                        loss_val = ptl.placement_policy_gradient(
                            q_metric_weights_np,
                            q_labels_np,
                            q_torch_scores,  # PyTorch tensor
                            n_samples=num_samples,
                            device=device,
                        )
                        current_batch_loss += loss_val
                    else:  # PL-Rank style losses (expect NumPy scores)
                        use_doc_weights_for_loss = True
                        q_np_scores = (
                            q_torch_scores.detach().cpu().numpy().squeeze()
                        )  # Convert to NumPy 1D array

                        if args.loss == "groupfairpl":
                            doc_weights = plr.Group_Fair_PL(  # Expects NumPy arrays
                                q_metric_weights_np,
                                q_labels_np,
                                q_gender_labels_np,
                                fairness_constraints,  # NumPy
                                q_np_scores,
                                n_samples=num_samples,
                                group_n_samples=1,  # This was hardcoded in PLRank.py
                            )
                        elif args.loss == "plrank3":
                            doc_weights = plr.PL_rank_3(  # Expects NumPy arrays
                                q_metric_weights_np,
                                q_labels_np,
                                q_np_scores,
                                n_samples=num_samples,
                            )
                        else:
                            raise NotImplementedError(f"Unknown loss {args.loss}")

                        batch_doc_weights_np[batch_ranges[i] : batch_ranges[i + 1]] = (
                            doc_weights
                        )
                    method_train_time += time.time() - last_method_train_time

            if use_doc_weights_for_loss:
                # Convert batch_doc_weights_np to tensor for loss calculation
                batch_doc_weights_torch = torch.from_numpy(
                    batch_doc_weights_np.astype(np.float32)
                ).to(device)
                # Loss is sum over all docs in batch: - sum(scores * weights)
                # batch_torch_scores is (total_docs, 1), batch_doc_weights_torch is (total_docs,)
                current_batch_loss = -torch.sum(
                    batch_torch_scores.squeeze() * batch_doc_weights_torch
                )

            if cur_batch_size > 0 and isinstance(
                current_batch_loss, torch.Tensor
            ):  # Ensure loss is a tensor
                # Average loss over queries in the batch if not using doc_weights,
                # or if doc_weights are already query-specific sums.
                # The original TF code sums losses from policygradient, so we do too.
                # For doc_weights, it's a single sum over all docs.
                # To be consistent, let's average if not doc_weights.
                if not use_doc_weights_for_loss:
                    current_batch_loss = current_batch_loss / cur_batch_size

                current_batch_loss.backward()  # Compute gradients
                optimizer.step()  # Update weights
                epoch_loss += (
                    current_batch_loss.item() * cur_batch_size
                )  # Accumulate loss item

            steps += cur_batch_size
            if dynamic_samples:  # Original dynamic sample logic
                float_num_samples = 10 + steps * add_per_step
                num_samples = min(int(np.round(float_num_samples)), max_num_samples)

        lr_scheduler.step()  # Update learning rate at the end of epoch

        if (
            epoch_i % 50 == 0 or epoch_i == config["n_epochs"]
        ):  # Evaluate every 50 epochs or last epoch
            if epoch_i == config["n_epochs"]:
                is_last_epoch = True

            total_train_time += time.time() - last_total_train_time

            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # No gradients needed for evaluation
                tick = time.time()
                # evl.compute_results expects NumPy scores
                # Get scores from PyTorch model and convert to NumPy
                val_scores_np = model(
                    torch.from_numpy(
                        data.validation.feature_matrix.astype(np.float32)
                    ).to(device)
                )
                val_scores_np = val_scores_np.detach().cpu().numpy().squeeze()

                valid_result = (
                    evl.compute_results_from_scores(  # Use from_scores variant
                        data.validation,
                        val_scores_np,  # Pass NumPy scores
                        metric_weights,
                        valid_labels,
                        ideal_valid_metrics,
                        config["num_eval_samples"],
                        valid_gender_labels,
                        fairness_constraints,
                        prefix_fairness_constraints,
                        postprocess_algorithms,
                        is_last_epoch,
                    )
                )

                tock = time.time()
                print(f"validation time:\t{tock - tick:.2f}s")

                test_scores_np = model(
                    torch.from_numpy(data.test.feature_matrix.astype(np.float32)).to(
                        device
                    )
                )
                test_scores_np = test_scores_np.detach().cpu().numpy().squeeze()

                test_result = (
                    evl.compute_results_from_scores(  # Use from_scores variant
                        data.test,
                        test_scores_np,  # Pass NumPy scores
                        metric_weights,
                        test_labels,
                        ideal_test_metrics,
                        config["num_eval_samples"],
                        test_gender_labels,
                        fairness_constraints,
                        prefix_fairness_constraints,
                        postprocess_algorithms,
                        is_last_epoch,
                    )
                )
                print(f"test time:\t\t{time.time() - tock:.2f}s")

            avg_epoch_loss = epoch_loss / n_queries if n_queries > 0 else 0.0
            print(
                "EPOCH: %07.2f TIME: %04d LOSS: %0.4f"
                " VALI: exp: %0.4f"
                " TEST: exp: %0.4f"
                % (
                    epoch_i,
                    # This time is for the loss computation part
                    method_train_time,
                    avg_epoch_loss,
                    # Assuming 'none' is first
                    valid_result[0]["normalized expectation"],
                    # Assuming 'none' is first
                    test_result[0]["normalized expectation"],
                )
            )

            cur_result = {
                "steps": steps,
                "epoch": epoch_i,
                "train_loss": avg_epoch_loss,
                "train time": method_train_time,
                "total time": total_train_time,  # This is wall clock time for training phases
                "validation result": valid_result,
                "test result": test_result,
                "num_samples": num_samples,
            }
            epoch_results.append(cur_result)
            last_total_train_time = time.time()  # Reset for next timing interval

    output = {
        "dataset": dataset_name,
        "fold number": config["fold_id"],
        "run name": args.loss.replace("_", " "),
        "loss": args.loss.replace("_", " "),
        "model hyperparameters": model_params,  # PyTorch model_params will include input_dim
        "epoch results": epoch_results,
        "number of samples": num_samples if not dynamic_samples else "dynamic",
        "number of evaluation samples": config["num_eval_samples"],
        "cutoff": config["cutoff"],
        "fairness constraints": fairness_constraints.tolist(),
        "prefix fairness constraints": prefix_fairness_constraints.tolist(),
    }
    # if dynamic_samples: # Already handled above
    #     output["number of samples"] = "dynamic"

    print(f"Writing results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
