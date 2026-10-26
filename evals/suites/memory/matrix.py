"""Register source-only memory combinations without changing frozen grading code."""

import os
from pathlib import Path
from typing import ClassVar

from openai import AsyncOpenAI

from evals.suites.memory import runner
from evals.suites.memory.combinations import (
    CombinationVariant,
    DenseClaimsVariant,
    RawClaimsDenseOpenVikingRRFVariant,
    RawClaimsDenseRRFVariant,
    RawClaimsRRFVariant,
    RawOpenVikingRRFVariant,
    RetrievalComponent,
)
from evals.suites.memory.decisions import DecisionsVariant, NativeDecisionsReranker
from evals.suites.memory.hindsight import HindsightChunksVariant, HindsightVariant
from evals.suites.memory.openviking import OpenVikingVariant
from evals.suites.memory.raw_hybrid import RawHybridVariant, RawLexicalVariant
from evals.suites.memory.types import MemoryVariant


def existing_prefix(workspace: Path, variant: str) -> Path:
    return workspace.parent.parent / variant / workspace.name


class HindsightConciseVariant(HindsightVariant):
    variant_name = "hindsight_concise"

    @staticmethod
    def _extraction_mode() -> str:
        return "concise"


class _Blend(CombinationVariant):
    component_names: ClassVar[tuple[str, ...]]

    def __init__(self, workspace: Path) -> None:
        super().__init__(
            workspace,
            components=[
                RetrievalComponent(name, component(name, workspace))
                for name in self.component_names
            ],
        )


class ClaimsOpenVikingVariant(_Blend):
    component_names = ("claims_dense", "openviking")


class RawHindsightConciseVariant(_Blend):
    component_names = ("raw_hybrid", "hindsight_concise")


class ClaimsHindsightConciseVariant(_Blend):
    component_names = ("claims_dense", "hindsight_concise")


class OpenVikingHindsightConciseVariant(_Blend):
    component_names = ("openviking", "hindsight_concise")


class RawClaimsHindsightConciseVariant(_Blend):
    component_names = ("raw_hybrid", "claims_dense", "hindsight_concise")


class RawOpenVikingHindsightConciseVariant(_Blend):
    component_names = ("raw_hybrid", "openviking", "hindsight_concise")


class ClaimsOpenVikingHindsightConciseVariant(_Blend):
    component_names = ("claims_dense", "openviking", "hindsight_concise")


class AllComponentsVariant(_Blend):
    component_names = ("raw_hybrid", "claims_dense", "openviking", "hindsight_concise")


class RawHindsightChunksVariant(_Blend):
    component_names = ("raw_hybrid", "hindsight_chunks")


def component(name: str, workspace: Path) -> MemoryVariant:
    path = existing_prefix(workspace, name)
    factories: dict[str, type] = {
        "raw_hybrid": RawHybridVariant,
        "raw_lexical": RawLexicalVariant,
        "claims_dense": DenseClaimsVariant,
        "raw_claims_rrf": RawClaimsRRFVariant,
        "raw_claims_dense_rrf": RawClaimsDenseRRFVariant,
        "openviking": OpenVikingVariant,
        "raw_openviking_rrf": RawOpenVikingRRFVariant,
        "raw_claims_dense_openviking_rrf": RawClaimsDenseOpenVikingRRFVariant,
        "hindsight_chunks": HindsightChunksVariant,
        "hindsight": HindsightVariant,
        "hindsight_concise": HindsightConciseVariant,
        "claims_openviking_rrf": ClaimsOpenVikingVariant,
        "raw_hindsight_concise_rrf": RawHindsightConciseVariant,
        "claims_hindsight_concise_rrf": ClaimsHindsightConciseVariant,
        "openviking_hindsight_concise_rrf": OpenVikingHindsightConciseVariant,
        "raw_claims_hindsight_concise_rrf": RawClaimsHindsightConciseVariant,
        "raw_openviking_hindsight_concise_rrf": RawOpenVikingHindsightConciseVariant,
        "claims_openviking_hindsight_concise_rrf": ClaimsOpenVikingHindsightConciseVariant,
        "all_components_rrf": AllComponentsVariant,
        "raw_hindsight_chunks_rrf": RawHindsightChunksVariant,
    }
    return factories[name](path)  # type: ignore[no-any-return]


class _Ranked(DecisionsVariant):
    component_name: ClassVar[str]
    final_hits: ClassVar[int] = 8

    def __init__(self, workspace: Path) -> None:
        self._ranking_client = AsyncOpenAI(timeout=90, max_retries=2)
        ranker = NativeDecisionsReranker(
            self._ranking_client,
            model=os.getenv("MEMORY_EVAL_RANK_MODEL", "gpt-6-luna"),
            trace_directory=workspace / "rank-traces",
        )
        super().__init__(
            component(self.component_name, workspace),
            ranker,
            workspace=workspace,
            final_limit=self.final_hits,
            pool_limit=100,
        )

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            await self._ranking_client.close()


class RawHybridDecisionsVariant(_Ranked):
    component_name = "raw_hybrid"


class RawLexicalDecisionsVariant(_Ranked):
    component_name = "raw_lexical"


class DenseClaimsDecisionsVariant(_Ranked):
    component_name = "claims_dense"


class RawClaimsDenseDecisionsVariant(_Ranked):
    component_name = "raw_claims_dense_rrf"


class OpenVikingDecisionsVariant(_Ranked):
    component_name = "openviking"
    final_hits = 5


class RawOpenVikingDecisionsVariant(_Ranked):
    component_name = "raw_openviking_rrf"


class RawClaimsDenseOpenVikingDecisionsVariant(_Ranked):
    component_name = "raw_claims_dense_openviking_rrf"


class HindsightChunksDecisionsVariant(_Ranked):
    component_name = "hindsight_chunks"


class HindsightDecisionsVariant(_Ranked):
    component_name = "hindsight"


class HindsightConciseDecisionsVariant(_Ranked):
    component_name = "hindsight_concise"


class ClaimsOpenVikingDecisionsVariant(_Ranked):
    component_name = "claims_openviking_rrf"


class RawHindsightConciseDecisionsVariant(_Ranked):
    component_name = "raw_hindsight_concise_rrf"


class ClaimsHindsightConciseDecisionsVariant(_Ranked):
    component_name = "claims_hindsight_concise_rrf"


class OpenVikingHindsightConciseDecisionsVariant(_Ranked):
    component_name = "openviking_hindsight_concise_rrf"


class RawClaimsHindsightConciseDecisionsVariant(_Ranked):
    component_name = "raw_claims_hindsight_concise_rrf"


class RawOpenVikingHindsightConciseDecisionsVariant(_Ranked):
    component_name = "raw_openviking_hindsight_concise_rrf"


class ClaimsOpenVikingHindsightConciseDecisionsVariant(_Ranked):
    component_name = "claims_openviking_hindsight_concise_rrf"


class AllComponentsDecisionsVariant(_Ranked):
    component_name = "all_components_rrf"


class RawHindsightChunksDecisionsVariant(_Ranked):
    component_name = "raw_hindsight_chunks_rrf"


def register() -> None:
    runner.VARIANTS.update({
        "openviking": ("openviking", "OpenVikingVariant"),
        "claims_dense": ("combinations", "DenseClaimsVariant"),
        "raw_claims_rrf": ("combinations", "RawClaimsRRFVariant"),
        "raw_claims_dense_rrf": ("combinations", "RawClaimsDenseRRFVariant"),
        "raw_openviking_rrf": ("combinations", "RawOpenVikingRRFVariant"),
        "raw_claims_dense_openviking_rrf": (
            "combinations",
            "RawClaimsDenseOpenVikingRRFVariant",
        ),
        "raw_hybrid_decisions": ("matrix", "RawHybridDecisionsVariant"),
        "raw_lexical_decisions": ("matrix", "RawLexicalDecisionsVariant"),
        "claims_dense_decisions": ("matrix", "DenseClaimsDecisionsVariant"),
        "raw_claims_dense_decisions": ("matrix", "RawClaimsDenseDecisionsVariant"),
        "openviking_decisions": ("matrix", "OpenVikingDecisionsVariant"),
        "raw_openviking_decisions": ("matrix", "RawOpenVikingDecisionsVariant"),
        "raw_claims_dense_openviking_decisions": (
            "matrix",
            "RawClaimsDenseOpenVikingDecisionsVariant",
        ),
        "hindsight_chunks_decisions": ("matrix", "HindsightChunksDecisionsVariant"),
        "hindsight_concise": ("matrix", "HindsightConciseVariant"),
        "hindsight_decisions": ("matrix", "HindsightDecisionsVariant"),
        "hindsight_concise_decisions": ("matrix", "HindsightConciseDecisionsVariant"),
        "claims_openviking_rrf": ("matrix", "ClaimsOpenVikingVariant"),
        "raw_hindsight_concise_rrf": ("matrix", "RawHindsightConciseVariant"),
        "claims_hindsight_concise_rrf": ("matrix", "ClaimsHindsightConciseVariant"),
        "openviking_hindsight_concise_rrf": (
            "matrix",
            "OpenVikingHindsightConciseVariant",
        ),
        "raw_claims_hindsight_concise_rrf": (
            "matrix",
            "RawClaimsHindsightConciseVariant",
        ),
        "raw_openviking_hindsight_concise_rrf": (
            "matrix",
            "RawOpenVikingHindsightConciseVariant",
        ),
        "claims_openviking_hindsight_concise_rrf": (
            "matrix",
            "ClaimsOpenVikingHindsightConciseVariant",
        ),
        "all_components_rrf": ("matrix", "AllComponentsVariant"),
        "raw_hindsight_chunks_rrf": ("matrix", "RawHindsightChunksVariant"),
        "claims_openviking_decisions": ("matrix", "ClaimsOpenVikingDecisionsVariant"),
        "raw_hindsight_concise_decisions": (
            "matrix",
            "RawHindsightConciseDecisionsVariant",
        ),
        "claims_hindsight_concise_decisions": (
            "matrix",
            "ClaimsHindsightConciseDecisionsVariant",
        ),
        "openviking_hindsight_concise_decisions": (
            "matrix",
            "OpenVikingHindsightConciseDecisionsVariant",
        ),
        "raw_claims_hindsight_concise_decisions": (
            "matrix",
            "RawClaimsHindsightConciseDecisionsVariant",
        ),
        "raw_openviking_hindsight_concise_decisions": (
            "matrix",
            "RawOpenVikingHindsightConciseDecisionsVariant",
        ),
        "claims_openviking_hindsight_concise_decisions": (
            "matrix",
            "ClaimsOpenVikingHindsightConciseDecisionsVariant",
        ),
        "all_components_decisions": ("matrix", "AllComponentsDecisionsVariant"),
        "raw_hindsight_chunks_decisions": (
            "matrix",
            "RawHindsightChunksDecisionsVariant",
        ),
    })


if __name__ == "__main__":
    register()
    runner.main()
