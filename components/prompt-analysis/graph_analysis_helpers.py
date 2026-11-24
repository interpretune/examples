"""Attribution-graph helpers for the attribution analysis notebook and its analysis points.

The two-hop adjacency read the notebook's analysis points report, and the ridgeline plot of a Neumann series trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch

if TYPE_CHECKING:
    from circuit_tracer.graph import Graph


def _resolve_adjacency(graph: Any, adjacency_matrix: torch.Tensor | None, adj_matrix_name: str) -> torch.Tensor:
    """The adjacency matrix to analyse: the one given, else the named attribute of ``graph``, else a refusal."""
    if adjacency_matrix is not None:
        return adjacency_matrix
    if graph is None:
        raise ValueError("Either 'graph' or 'adjacency_matrix' must be provided.")
    return getattr(graph, adj_matrix_name)


def get_topk_2nd_order_adjacency(
    k: int,
    limit_node: int,
    inspect_logit_idxs: torch.Tensor,
    n_logits: int = 10,
    graph: Graph | None = None,
    adjacency_matrix: torch.Tensor | None = None,
    adj_matrix_name: str = "adjacency_matrix",
):
    """Returns the top-k 2nd order adjacency values and indices for non-error, non-logit nodes.

    Args:
        k (int): Number of top values to select per row.
        limit_node (int): Only consider nodes with indices < limit_node.
        top_adj_edge_postnorm_indices (torch.Tensor): Indices of top adjacency edges (shape: [n_rows, k]).
        graph (Graph, optional): Graph object containing adjacency matrices.
        adjacency_matrix (torch.Tensor, optional): Adjacency matrix to use directly.
        adj_matrix_name (str): Name of the adjacency matrix attribute in graph.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (topk_values, topk_indices), both of shape [n_rows, k, k]
    """
    adjacency_matrix = _resolve_adjacency(graph, adjacency_matrix, adj_matrix_name)
    adj_logit_idxs = (adjacency_matrix.shape[0] - n_logits) + inspect_logit_idxs
    first_order_values, first_order_indices = torch.topk(adjacency_matrix[adj_logit_idxs, :], k)
    adj_mask = first_order_indices < limit_node
    safe_indices = first_order_indices.clone()
    safe_indices[~adj_mask] = 0  # Replace invalid indices with a dummy row (e.g., 0)
    selected_rows = adjacency_matrix[safe_indices]
    selected_rows[~adj_mask] = float("-inf")  # Mask out invalid rows
    flat_rows = selected_rows.view(-1, adjacency_matrix.shape[1])
    topk_values, topk_indices = torch.topk(flat_rows, k, dim=1)
    topk_values = topk_values.view(*selected_rows.shape[:2], k)
    topk_indices = topk_indices.view(*selected_rows.shape[:2], k)
    return topk_values, topk_indices, first_order_values, first_order_indices


def tensor_distribution_summary_stats(tensor: torch.Tensor):
    """Returns summary statistics (mean, std, min, max, median, nonzero count) for each row in a tensor. Each row
    is assumed to represent a distribution from a given iteration.

    Args:
        tensor (torch.Tensor): shape (num_iterations, num_values)

    Returns:
        list[dict]: List of dicts containing summary stats for each iteration.
    """
    stats = []
    for i in range(tensor.shape[0]):
        row = tensor[i]
        stats.append(
            {
                "iteration": i,
                "mean": row.mean().item(),
                "std": row.std().item(),
                "min": row.min().item(),
                "max": row.max().item(),
                "median": row.median().item(),
                "nonzero_count": (row != 0).sum().item(),
                "total_count": row.numel(),
            }
        )
    return stats


def plot_ridgeline_convergence(
    data: torch.Tensor,
    stats: list[dict[str, Any]] | None = None,
    title: str = "Convergence Distribution Ridgeline Plot",
):
    """Plot ridgeline distributions for convergence data using Plotly.

    Args:
        data: Tensor of shape [n_iterations_to_convergence, n_total_nodes] containing the convergence data
        stats: List of dictionaries containing summary statistics for each iteration
        title: Title for the plot
    """
    if not stats:
        stats = tensor_distribution_summary_stats(data)
    if data.shape[0] != len(stats):
        raise ValueError(f"Data has {data.shape[0]} rows but {len(stats)} stat entries")

    fig = go.Figure()

    # Color scale for iterations
    colors = px.colors.sequential.Viridis
    color_scale = np.linspace(0, 1, len(stats))

    # Calculate y-positions for ridgeline effect
    y_spacing = 1.0
    y_positions = np.arange(len(stats)) * y_spacing

    # Process each iteration
    for i, (row_data, stat) in enumerate(zip(data, stats)):
        # Convert to numpy and filter out zeros for better visualization
        values = row_data.cpu().numpy() if torch.is_tensor(row_data) else row_data
        nonzero_values = values[values > 0]

        if len(nonzero_values) == 0:
            # Handle the case where all values are zero (like iteration 27)
            continue

        # Use log scale for better visualization of wide range
        log_values = np.log10(nonzero_values + 1e-40)  # Add small epsilon to handle zeros

        # Create histogram to get density
        counts, bin_edges = np.histogram(log_values, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize density for ridgeline effect (scale down for visibility)
        normalized_density = counts / np.max(counts) * 0.8 if np.max(counts) > 0 else counts

        # Create y-coordinates for this distribution
        y_coords = y_positions[i] + normalized_density

        # Add the distribution curve
        color_idx = int(color_scale[i] * (len(colors) - 1))
        color = colors[color_idx]

        # Fill area under curve
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([bin_centers, bin_centers[::-1]]),
                y=np.concatenate([y_coords, np.full(len(bin_centers), y_positions[i])]),
                fill="toself",
                fillcolor=color,
                opacity=0.6,
                line=dict(color=color, width=1),
                name=f"Iteration {stat['iteration']}",
                hovertemplate=(
                    f"Iteration {stat['iteration']}<br>"
                    + f"Mean: {stat['mean']:.2e}<br>"
                    + f"Max: {stat['max']:.2e}<br>"
                    + f"Nonzero: {stat['nonzero_count']}/{stat['total_count']}<br>"
                    + "Log10(Value): %{x:.2f}<br>"
                    + "<extra></extra>"
                ),
            )
        )

        # Add baseline
        fig.add_trace(
            go.Scatter(
                x=[bin_centers.min(), bin_centers.max()],
                y=[y_positions[i], y_positions[i]],
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title="Log10(Value + ε)", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(
            title="Iteration",
            tickmode="array",
            tickvals=y_positions,
            ticktext=[f"Iter {i}" for i in range(len(stats))],
            showgrid=False,
        ),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        # sized by its container rather than a fixed width, so the figure fits a docs page as well as a notebook
        autosize=True,
        height=800,
        plot_bgcolor="white",
    )

    # Add annotations for key statistics (mean and max for first several iterations)
    for i, stat in enumerate(stats[:8]):  # Show stats for first 8 iterations
        if stat["nonzero_count"] > 0:
            # Mean annotation
            fig.add_annotation(
                x=np.log10(stat["mean"] + 1e-40),
                y=y_positions[i] + 0.3,
                text=f"μ={stat['mean']:.1e}",
                showarrow=False,
                font=dict(size=8, color="blue"),
                bgcolor="white",
                bordercolor="blue",
                borderwidth=1,
                opacity=0.9,
            )

            # Max annotation
            fig.add_annotation(
                x=np.log10(stat["max"] + 1e-40),
                y=y_positions[i] + 0.6,
                text=f"max={stat['max']:.1e}",
                showarrow=False,
                font=dict(size=8, color="red"),
                bgcolor="white",
                bordercolor="red",
                borderwidth=1,
                opacity=0.9,
            )

    return fig
