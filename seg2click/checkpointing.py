"""Independent vision/LLM checkpointing for the Seg2Click Qwen2-VL / Qwen2.5-VL wrappers."""

from __future__ import annotations

from typing import Any

from seg2click.qwen_vl_model import Seg2ClickQwen25VLWrapper, Seg2ClickQwen2VL2BWrapper


class _SplitCheckpointing:
    """Retain the requested branch policy when Trainer enables checkpointing.

    This changes activation recomputation only, not weights or decoder artifacts.
    Non-reentrant checkpointing supports the two vision forwards under DDP.
    """

    def __init__(
        self,
        base_model: Any,
        *,
        vision_gradient_checkpointing: bool = True,
        llm_gradient_checkpointing: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_model, **kwargs)
        if not isinstance(vision_gradient_checkpointing, bool) or not isinstance(
            llm_gradient_checkpointing, bool
        ):
            raise TypeError("Checkpointing switches must be booleans")
        self.vision_gradient_checkpointing = vision_gradient_checkpointing
        self.llm_gradient_checkpointing = llm_gradient_checkpointing
        self.gradient_checkpointing_enable()

    def _checkpointing_branches(self) -> tuple:
        native_model = self._vision_model()
        return (
            ("vision", native_model.visual, self.vision_gradient_checkpointing),
            ("llm", native_model.language_model, self.llm_gradient_checkpointing),
        )

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict | None = None,
        every_n_layers: int = 1,
    ) -> None:
        options = dict(gradient_checkpointing_kwargs or {})
        options.setdefault("use_reentrant", False)
        if options["use_reentrant"] is not False:
            raise ValueError("Seg2Click split checkpointing requires use_reentrant=False")
        if every_n_layers != 1:
            raise ValueError("Split checkpointing supports whole branches only (every_n_layers=1)")
        for _, branch, enabled in self._checkpointing_branches():
            # HF enable_input_require_grads is not idempotent: remove the hooks
            # owned by this branch before Trainer reapplies the same policy.
            branch.disable_input_require_grads()
            if enabled:
                branch.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=options,
                    every_n_layers=1,
                )
            else:
                branch.gradient_checkpointing_disable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable both temporarily; enable() restores the configured policy."""
        for _, branch, _ in self._checkpointing_branches():
            branch.gradient_checkpointing_disable()
            branch.disable_input_require_grads()

    def checkpointing_state(self) -> dict[str, dict[str, Any]]:
        """Return effective block flags for provenance and regression tests."""
        result = {}
        for name, branch, requested in self._checkpointing_branches():
            blocks = branch.blocks if name == "vision" else branch.layers
            flags = [bool(block.gradient_checkpointing) for block in blocks]
            result[name] = {
                "requested": requested,
                "enabled_blocks": sum(flags),
                "total_blocks": len(flags),
                "branch_enabled": bool(branch.gradient_checkpointing),
            }
        return result


class Seg2ClickQwen25VLSplitCheckpointingWrapper(_SplitCheckpointing, Seg2ClickQwen25VLWrapper):
    """Qwen2.5-VL with independently checkpointed vision and language branches."""


class Seg2ClickQwen2VL2BSplitCheckpointingWrapper(_SplitCheckpointing, Seg2ClickQwen2VL2BWrapper):
    """Qwen2-VL with independently checkpointed vision and language branches."""


__all__ = [
    "Seg2ClickQwen25VLSplitCheckpointingWrapper",
    "Seg2ClickQwen2VL2BSplitCheckpointingWrapper",
]
