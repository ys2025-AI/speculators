import logging
from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.metrics import LossConfig, kl_div_loss, resolve_loss_config
from speculators.models.utils import conditional_torch_compile

logger = logging.getLogger(__name__)

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DSparkDraftModel",
]


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a Markov logit-bias head and a confidence head.

    After the base draft logits are produced, the Markov head biases position
    ``k`` using the previous block token and the confidence head predicts each
    position's acceptance probability. Everything else is inherited from DFlash.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size

        self.markov_head: MarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires markov_rank > 0."
                )
            input_dim = hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(input_dim)

        # Option A': residual draft adapter on the lm_head path. Zero-init the
        # last layer so it starts as identity (== DFlash behavior); only learns
        # the delta to realign draft logits (e.g. to sample_from_anchor=true)
        # while the decoder + lm_head stay frozen (use with --freeze-backbone).
        if config.draft_adapter_rank > 0:
            r = config.draft_adapter_rank
            self.draft_adapter = nn.Sequential(
                nn.Linear(hidden_size, r, bias=False),
                nn.GELU(),
                nn.Linear(r, hidden_size, bias=False),
            )
            nn.init.zeros_(self.draft_adapter[-1].weight)  # type: ignore[index]

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        """Create a DSpark model from training arguments (mirrors DFlash)."""
        enable_confidence_head_arg = kwargs.get("enable_confidence_head")
        confidence_head_with_markov_arg = kwargs.get("confidence_head_with_markov")
        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True
                if enable_confidence_head_arg is None
                else enable_confidence_head_arg
            ),
            confidence_head_with_markov=(
                True
                if confidence_head_with_markov_arg is None
                else confidence_head_with_markov_arg
            ),
            draft_adapter_rank=kwargs.get("draft_adapter_rank", 0),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()

        init_from_dflash = kwargs.get("init_from_dflash")
        if init_from_dflash:
            model.load_dflash_backbone(init_from_dflash)

            freeze_backbone = kwargs.get("freeze_backbone", False)
            if freeze_backbone:
                head_hints = ("markov_head", "confidence_head", "draft_adapter")
                frozen_count = 0
                trainable_count = 0
                for name, param in model.named_parameters():
                    if any(h in name for h in head_hints):
                        param.requires_grad = True
                        trainable_count += 1
                    else:
                        param.requires_grad = False
                        frozen_count += 1
                logger.info(
                    "Frozen backbone: %d params frozen, %d head params trainable",
                    frozen_count,
                    trainable_count,
                )
        return model

    # State-dict keys excluded when borrowing a DFlash backbone: they are
    # verifier-derived (embed_tokens/lm_head/verifier_norm, set by
    # load_verifier_weights) or vocab-mapping buffers (t2d/d2t, set by
    # load_vocab_mappings). Preserving the DSpark model's own values keeps the
    # draft-vocab mapping correct and avoids clobbering frozen verifier weights.
    _DFLASH_BACKBONE_SKIP_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "embed_tokens.weight",
            "lm_head.weight",
            "verifier_lm_head.weight",
            "verifier_norm.weight",
            "t2d",
            "d2t",
        }
    )

    def load_dflash_backbone(self, dflash_path: str) -> None:
        """Initialize the DSpark draft body from a trained DFlash speculator.

        DSpark inherits DFlash's decoder (layers / fc / hidden_norm / norm)
        unchanged, so every backbone tensor transfers by name. The Markov and
        confidence heads -- absent from any DFlash checkpoint -- keep their
        random ``__init__`` values; verifier-derived weights and vocab-mapping
        buffers are preserved.
        """
        dflash_model = DFlashDraftModel.from_pretrained(dflash_path)
        src_sd = dflash_model.state_dict()
        del dflash_model

        backbone_sd = {
            k: v for k, v in src_sd.items() if k not in self._DFLASH_BACKBONE_SKIP_KEYS
        }
        missing, unexpected = self.load_state_dict(backbone_sd, strict=False)
        if unexpected:
            raise ValueError(
                f"--init-from-dflash: DFlash checkpoint '{dflash_path}' contains "
                f"weights absent from the DSpark model: {unexpected}"
            )
        # `missing` are the DSpark-only Markov/confidence head tensors plus the
        # skipped verifier/vocab keys; they stay at their random / already-loaded
        # values.
        logger.info(
            "--init-from-dflash: loaded %d backbone tensors from '%s' "
            "(%d DSpark-only tensors left at their current init).",
            len(backbone_sd),
            dflash_path,
            len(missing),
        )

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "confidence_head_alpha": confidence_head_alpha,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        return dict(shared), dict(shared)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )

        # DSpark: add the Markov logit bias and predict per-position confidence.
        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        # Ground-truth block tokens (verifier vocab); position 0 is the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        if self.config.sample_from_anchor:
            # With sample_from_anchor=True (DSpark default), slot k predicts
            # token p+k+1 and the inference Markov chain conditions slot k's
            # bias on the token at the previous position p+k.
            prev_token_ids = block_tokens
        else:
            # With sample_from_anchor=False (Dflash default), slot k predicts
            # token p+k, so the previous token within the block is
            # block_tokens[:, k-1] (shifted).
            prev_token_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )  # [num_blocks, block]
        hidden_blocks = hidden.view(num_blocks, block, -1)

        confidence_logits = None
        prev_emb = None
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )

        if self.confidence_head is not None:
            # confidence_head_with_markov requires markov_rank > 0 (enforced in
            # __init__), so prev_emb is always set when the flag is on.
            if self.config.confidence_head_with_markov and prev_emb is not None:
                conf_features = torch.cat(
                    [hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1
                )
            else:
                conf_features = hidden_blocks
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )

        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        return None, loss, metrics
