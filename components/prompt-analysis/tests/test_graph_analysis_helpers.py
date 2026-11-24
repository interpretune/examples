"""The attribution analysis notebook's graph helpers on small hand-built matrices."""

from __future__ import annotations

from types import SimpleNamespace

import graph_analysis_helpers as gah
import pytest
import torch

# rows 0 and 1 are features, 2 an error node, 3 and 4 the logits
ADJACENCY = torch.tensor(
    [
        [0.0, 1.0, 2.0, 0.0, 0.0],
        [3.0, 0.0, 4.0, 0.0, 0.0],
        [9.0, 9.0, 9.0, 0.0, 0.0],
        [5.0, 1.0, 7.0, 0.0, 0.0],
        [2.0, 8.0, 0.0, 0.0, 0.0],
    ]
)


def test_second_order_follows_only_nodes_below_the_limit():
    values, indices, first_values, first_indices = gah.get_topk_2nd_order_adjacency(
        2, limit_node=2, inspect_logit_idxs=torch.tensor([0, 1]), n_logits=2, adjacency_matrix=ADJACENCY
    )
    assert first_indices.tolist() == [[2, 0], [1, 0]] and first_values.tolist() == [[7.0, 5.0], [8.0, 2.0]]
    assert bool(torch.isinf(values[0, 0]).all())  # the error node's row is masked, not followed
    assert values[0, 1].tolist() == [2.0, 1.0] and indices[0, 1].tolist() == [2, 1]
    assert values[1, 0].tolist() == [4.0, 3.0] and indices[1, 0].tolist() == [2, 0]


def test_the_adjacency_comes_from_the_graph_when_none_is_given():
    args = {"limit_node": 2, "inspect_logit_idxs": torch.tensor([0]), "n_logits": 2}
    given = gah.get_topk_2nd_order_adjacency(1, adjacency_matrix=ADJACENCY, **args)
    read = gah.get_topk_2nd_order_adjacency(1, graph=SimpleNamespace(adjacency_matrix=ADJACENCY), **args)
    assert all(torch.equal(a, b) for a, b in zip(given, read))
    with pytest.raises(ValueError, match="Either 'graph' or 'adjacency_matrix'"):
        gah.get_topk_2nd_order_adjacency(1, **args)


def test_the_ridgeline_draws_one_curve_per_iteration_with_mass():
    trace = torch.tensor([[0.5, 0.25, 0.0, 0.1], [0.05, 0.01, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    stats = gah.tensor_distribution_summary_stats(trace)
    assert [s["nonzero_count"] for s in stats] == [3, 2, 0] and stats[1]["max"] == pytest.approx(0.05)
    fig = gah.plot_ridgeline_convergence(trace, title="trace")
    assert [t.name for t in fig.data if t.fill == "toself"] == ["Iteration 0", "Iteration 1"]
    with pytest.raises(ValueError, match="3 rows but 1 stat entries"):
        gah.plot_ridgeline_convergence(trace, stats=stats[:1])
