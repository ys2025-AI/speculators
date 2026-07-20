"""Tests for the DSpark ``--init-from-dflash`` backbone initialization."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _qwen3():
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


def _spec_cfg(algo):
    return SpeculatorsConfig(
        algorithm=algo,
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(name_or_path="dummy", architectures=[]),
    )


def _dflash_cfg():
    return DFlashSpeculatorConfig(
        transformer_layer_config=_qwen3(),
        draft_vocab_size=32,
        block_size=4,
        aux_hidden_state_layer_ids=[0],
        mask_token_id=1,
        speculators_config=_spec_cfg("dflash"),
    )


def _dspark_cfg():
    return DSparkSpeculatorConfig(
        transformer_layer_config=_qwen3(),
        draft_vocab_size=32,
        block_size=4,
        aux_hidden_state_layer_ids=[0],
        mask_token_id=1,
        markov_rank=8,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        speculators_config=_spec_cfg("dspark"),
    )


def _build_dspark():
    # Stub out verifier download; embed_tokens/lm_head/verifier_* stay NaN.
    with patch.object(DSparkDraftModel, "load_verifier_weights"):
        return DSparkDraftModel(_dspark_cfg())


def _build_dflash():
    with patch.object(DFlashDraftModel, "load_verifier_weights"):
        return DFlashDraftModel(_dflash_cfg())


def test_load_dflash_backbone_transfers_body_keeps_heads():
    dspark = _build_dspark()
    dflash = _build_dflash()  # same backbone shape, independently initialized

    # Sanity: before load, the backbone differs between the two random inits.
    assert not torch.equal(dspark.fc.weight, dflash.fc.weight)

    # Give the DFlash source a non-NaN embed_tokens so we can confirm it is
    # *skipped* (the DSpark model's own value must be preserved, not clobbered).
    with torch.no_grad():
        dflash.embed_tokens.weight.fill_(1.0)
    assert dspark.embed_tokens.weight.isnan().all()

    with patch.object(DFlashDraftModel, "from_pretrained", return_value=dflash):
        dspark.load_dflash_backbone("fake/dflash/path")

    # Backbone weights transferred from the DFlash source.
    assert torch.equal(dspark.fc.weight, dflash.fc.weight)
    assert torch.equal(dspark.norm.weight, dflash.norm.weight)
    assert torch.equal(
        dspark.layers[0].self_attn.q_proj.weight,
        dflash.layers[0].self_attn.q_proj.weight,
    )
    assert torch.equal(dspark.hidden_norm.weight, dflash.hidden_norm.weight)

    # Verifier-derived embed_tokens was skipped: still NaN, not overwritten by
    # the DFlash source's all-ones tensor.
    assert dspark.embed_tokens.weight.isnan().all()

    # DSpark-only heads are present and finite (random init, not NaN).
    assert dspark.markov_head is not None
    assert not dspark.markov_head.markov_w1.weight.isnan().any()
    assert not dspark.markov_head.markov_w2.weight.isnan().any()
    assert dspark.confidence_head is not None
    assert not dspark.confidence_head.proj.weight.isnan().any()
    assert not dspark.confidence_head.proj.bias.isnan().any()


def test_load_dflash_backbone_raises_on_unexpected_keys():
    dspark = _build_dspark()
    dflash = _build_dflash()

    # Inject a key that doesn't exist on DSpark -> must be flagged as unexpected.
    src_sd = dflash.state_dict()
    src_sd["bogus.not.in.dspark"] = torch.zeros(1)

    fake = SimpleNamespace(state_dict=lambda: src_sd)

    with (
        patch.object(DFlashDraftModel, "from_pretrained", return_value=fake),
        pytest.raises(ValueError, match="weights absent from the DSpark model"),
    ):
        dspark.load_dflash_backbone("fake/dflash/path")


def test_from_training_args_invokes_load_dflash_backbone():
    """When ``init_from_dflash`` is passed in kwargs, from_training_args calls
    load_dflash_backbone after the standard verifier-weight load."""
    base_kwargs = {
        "transformer_layer_config": _qwen3(),
        "draft_vocab_size": 32,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [0],
        "mask_token_id": 1,
        "sliding_window_non_causal": False,
        "speculators_config": _spec_cfg("dspark"),
    }
    with (
        patch.object(
            DFlashDraftModel,
            "_build_base_config_kwargs",
            return_value=base_kwargs,
        ),
        patch.object(DSparkDraftModel, "load_verifier_weights"),
        patch.object(DSparkDraftModel, "load_dflash_backbone") as mock_load,
    ):
        DSparkDraftModel.from_training_args(
            verifier_config=_qwen3(),
            t2d=None,
            d2t=None,
            init_from_dflash="some/dflash/ckpt",
            draft_vocab_size=32,
            block_size=4,
            verifier_name_or_path="dummy",
        )

    mock_load.assert_called_once_with("some/dflash/ckpt")


def test_from_training_args_skips_load_when_init_from_dflash_empty():
    """An empty/None ``init_from_dflash`` must not call load_dflash_backbone."""
    base_kwargs = {
        "transformer_layer_config": _qwen3(),
        "draft_vocab_size": 32,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [0],
        "mask_token_id": 1,
        "sliding_window_non_causal": False,
        "speculators_config": _spec_cfg("dspark"),
    }
    with (
        patch.object(
            DFlashDraftModel,
            "_build_base_config_kwargs",
            return_value=base_kwargs,
        ),
        patch.object(DSparkDraftModel, "load_verifier_weights"),
        patch.object(DSparkDraftModel, "load_dflash_backbone") as mock_load,
    ):
        DSparkDraftModel.from_training_args(
            verifier_config=_qwen3(),
            t2d=None,
            d2t=None,
            init_from_dflash="",  # argparse default
            draft_vocab_size=32,
            block_size=4,
            verifier_name_or_path="dummy",
        )

    mock_load.assert_not_called()
