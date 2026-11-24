"""The attribution analysis notebook's target-token record and shape report."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from attribution_analysis_helpers import TargetTokenAnalysis, collect_shapes

from interpretune.analysis.injection.orchestrator import VarAnnotate


class _Tokenizer:
    vocab = {"Dallas": 3, "Austin": 5}

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[t] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [next(t for t, i in self.vocab.items() if i == x) for x in ids]


def test_tokens_and_ids_are_derived_from_each_other():
    by_token = TargetTokenAnalysis(tokens=["Dallas", "Austin"], tokenizer=_Tokenizer())
    assert by_token.token_ids is not None and by_token.token_ids.tolist() == [3, 5]
    by_id = TargetTokenAnalysis(token_ids=torch.tensor([5, 3]), tokenizer=_Tokenizer())
    assert by_id.tokens == ["Austin", "Dallas"]


def test_an_underspecified_or_misaligned_record_is_refused():
    with pytest.raises(ValueError, match="At least one of 'tokens' or 'token_ids'"):
        TargetTokenAnalysis()
    with pytest.raises(ValueError, match="when no tokenizer is available"):
        TargetTokenAnalysis(tokens=["Dallas"])
    with pytest.raises(ValueError, match="must match tokens length 1"):
        TargetTokenAnalysis(tokens=["Dallas"], token_ids=torch.tensor([3]), logit_indices=torch.tensor([0, 1]))


def test_update_logit_info_realigns_every_field_to_the_logit_order():
    record = TargetTokenAnalysis(tokens=["Dallas", "Austin"], tokenizer=_Tokenizer())
    record.update_logit_info(torch.tensor([5, 9, 3]), torch.tensor([0.5, 0.3, 0.2]))
    assert record.tokens == ["Austin", "Dallas"]
    assert record.token_ids is not None and record.token_ids.tolist() == [5, 3]
    assert record.logit_indices is not None and record.logit_indices.tolist() == [0, 2]
    assert record.logit_probabilities is not None
    assert record.logit_probabilities.tolist() == pytest.approx([0.5, 0.2])
    with pytest.raises(ValueError, match="No matching token_ids"):
        record.update_logit_info(torch.tensor([7]), torch.tensor([1.0]))


def test_collect_shapes_reports_a_shape_or_says_it_found_none():
    data = collect_shapes({}, {"ctx": SimpleNamespace(m=torch.zeros(2, 3))}, ["ctx.m", VarAnnotate("ctx.absent")])
    assert data["ctx.m.shape"].output == "(2, 3)"
    assert data["ctx.absent.shape"].output == "not found or does not have shape attribute"
