from operator import itemgetter

import pytest
from dirty_equals import IsList, IsStr

from dashulya.memory.hybrid import embedding_texts, query_terms
from dashulya.memory.ranking import reciprocal_rank_fusion, select_source_diverse
from tests.originals import original


class TestRanking:
    def test_rrf_uses_ordinal_ranks_and_deterministic_explicit_ties(self) -> None:
        fused = reciprocal_rank_fusion([[2, 1], [1, 3]], tie_order={1: 0, 2: 1, 3: 2})
        assert [key for key, _ in fused] == [1, 2, 3]
        assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
        assert reciprocal_rank_fusion([[1, 1], [2]], tie_order={1: 1, 2: 0})[0][0] == 2
        with pytest.raises(ValueError, match="tie position"):
            reciprocal_rank_fusion([[1]], tie_order={})

    def test_source_diversity_suppresses_covered_anchor_but_not_new_reply(self) -> None:
        candidates = [("a", ("parent", "a")), ("parent", ("parent",)), ("b", ("a", "b"))]
        selected = select_source_diverse(
            candidates, anchor=itemgetter(0), sources=itemgetter(1), limit=8
        )
        assert [item[0] for item in selected] == ["a", "b"]
        assert (
            select_source_diverse(
                candidates, anchor=itemgetter(0), sources=itemgetter(1), limit=0
            )
            == []
        )

    def test_composite_source_keys_preserve_cross_chat_and_revision_identity(
        self,
    ) -> None:
        first = ("group", 1, 1)
        other_chat = ("private", 1, 1)
        other_revision = ("group", 1, 2)
        result = reciprocal_rank_fusion(
            [[first, other_chat], [other_revision]],
            tie_order={first: 0, other_chat: 1, other_revision: 2},
        )
        assert {key for key, _ in result} == {first, other_chat, other_revision}

    def test_query_terms_are_normalized_deduplicated_and_bounded(self) -> None:
        assert query_terms("и THE инструментами BB bb") == ["инструмент", "bb"]
        assert len(query_terms(" ".join(f"term{i}" for i in range(60)))) == 48

    def test_embedding_payloads_keep_each_source_and_bound_reply_chains(self) -> None:
        texts = embedding_texts([
            original(3, "z" * 20000, parent=2),
            original(1, "x" * 20000),
            original(2, "y" * 20000, parent=1),
        ])
        assert texts == IsList(IsStr(min_length=1, max_length=12000), length=3)
        assert [len(text) for text in texts] == [4000, 8002, 12000]
        assert texts[1][:4002] == texts[0] + "\n\n"
        assert texts[2][:8004] == texts[1] + "\n\n"
        assert "[message_id=2 " in texts[1][4002:]
        assert "[message_id=3 " in texts[2][8004:]

        assert embedding_texts([]) == []
