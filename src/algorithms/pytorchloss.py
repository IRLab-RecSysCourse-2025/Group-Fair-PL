# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import torch
import numpy as np
from ..utils import plackettluce as pl  # This uses NumPy, so its output will be NumPy


def placement_policy_gradient(
    rank_weights, labels, scores, n_samples=None, sampled_rankings_np=None, device="cpu"
):
    # rank_weights, labels are NumPy arrays
    # scores is a PyTorch tensor (output from the model)
    # sampled_rankings_np is a NumPy array

    n_docs = labels.shape[0]
    cutoff = min(rank_weights.shape[0], n_docs)

    # Convert NumPy inputs to PyTorch tensors
    rank_weights_torch = torch.from_numpy(rank_weights.astype(np.float32)).to(device)
    labels_torch = torch.from_numpy(labels.astype(np.float32)).to(device)

    if sampled_rankings_np is None:
        # pl.gumbel_sample_rankings expects NumPy scores
        np_scores = (
            scores.detach().cpu().numpy().squeeze()
        )  # Squeeze if scores is (N,1)
        sampled_rankings_np = pl.gumbel_sample_rankings(
            np_scores, n_samples, cutoff=cutoff, return_full_rankings=True
        )[0]
    else:
        n_samples = sampled_rankings_np.shape[0]

    # sampled_rankings_np is (n_samples, n_docs_in_ranking)
    # scores is (total_docs_in_query, 1) or (total_docs_in_query,)
    # We need to gather scores based on sampled_rankings_np
    # Ensure scores is 1D for easier gathering if it was (N,1)
    scores_1d = scores.squeeze()

    # Create tensor from sampled_rankings_np for gathering
    sampled_rankings_torch = torch.from_numpy(sampled_rankings_np.astype(np.int64)).to(
        device
    )

    # Gather scores for each document in each sampled ranking
    # sampled_scores will be (n_samples, n_docs_in_ranking)
    sampled_scores = scores_1d[sampled_rankings_torch]

    # cumulative_logsumexp in PyTorch
    # For reverse=True: flip, logcumsumexp, flip back
    denom = sampled_scores.flip(dims=[1]).logcumsumexp(dim=1).flip(dims=[1])

    sample_log_prob = sampled_scores[:, :cutoff] - denom[:, :cutoff]

    # Gather labels for each document in each sampled ranking
    # labels_torch is 1D (total_docs_in_query,)
    sampled_labels_torch = labels_torch[sampled_rankings_torch[:, :cutoff]]

    rewards = rank_weights_torch[None, :cutoff] * sampled_labels_torch
    cum_rewards = torch.cumsum(rewards, dim=1, reverse=True)

    loss = torch.sum(torch.mean(sample_log_prob * cum_rewards, dim=0))
    return -loss


def policy_gradient(
    rank_weights, labels, scores, n_samples=None, sampled_rankings_np=None, device="cpu"
):
    # rank_weights, labels are NumPy arrays
    # scores is a PyTorch tensor
    # sampled_rankings_np is a NumPy array

    n_docs = labels.shape[0]
    cutoff = min(rank_weights.shape[0], n_docs)

    # Convert NumPy inputs to PyTorch tensors
    rank_weights_torch = torch.from_numpy(rank_weights.astype(np.float32)).to(device)
    labels_torch = torch.from_numpy(labels.astype(np.float32)).to(device)

    if sampled_rankings_np is None:
        np_scores = scores.detach().cpu().numpy().squeeze()
        sampled_rankings_np = pl.gumbel_sample_rankings(
            np_scores, n_samples, cutoff=cutoff, return_full_rankings=True
        )[0]
    else:
        n_samples = sampled_rankings_np.shape[0]

    scores_1d = scores.squeeze()
    sampled_rankings_torch = torch.from_numpy(sampled_rankings_np.astype(np.int64)).to(
        device
    )
    sampled_scores = scores_1d[sampled_rankings_torch]

    denom = sampled_scores.flip(dims=[1]).logcumsumexp(dim=1).flip(dims=[1])
    sample_log_prob = sampled_scores[:, :cutoff] - denom[:, :cutoff]
    final_prob_loss = torch.sum(sample_log_prob, dim=1)

    sampled_labels_torch = labels_torch[sampled_rankings_torch[:, :cutoff]]
    rewards_np = torch.sum(
        rank_weights_torch[None, :cutoff] * sampled_labels_torch, dim=1
    )

    loss = torch.mean(final_prob_loss * rewards_np, dim=0)
    return -loss
