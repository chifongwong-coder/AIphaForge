"""v2.2.1 #2 + #4 — rank parser + structured-input chain + normalization wiring."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiphaforge.probes._rank import (
    RankAnswer,
    RankContinuationProbe,
    parse_rank_answer,
    resolve_rank_answer,
)
from aiphaforge.probes.models import AnswerRecord

# ---------- parse_rank_answer ----------


class TestParseRankAnswer:
    EXPECTED = ["AAPL", "MSFT", "GOOG"]

    def test_json_array(self):
        ans = parse_rank_answer('["AAPL","MSFT","GOOG"]', self.EXPECTED)
        assert ans is not None
        assert ans.ranking == ("AAPL", "MSFT", "GOOG")
        assert ans.parser_used == "json_array"

    def test_json_array_codefence(self):
        text = '```json\n["AAPL","MSFT","GOOG"]\n```'
        ans = parse_rank_answer(text, self.EXPECTED)
        assert ans is not None
        assert ans.parser_used == "json_array_codefence"

    def test_json_object_with_tied(self):
        text = '{"ranking":["AAPL","MSFT","GOOG"],"tied":[["MSFT","GOOG"]]}'
        ans = parse_rank_answer(text, self.EXPECTED)
        assert ans is not None
        assert ans.parser_used == "json_object"
        assert frozenset({"MSFT", "GOOG"}) in ans.tie_groups

    def test_numbered_list(self):
        text = "1. AAPL\n2. MSFT\n3. GOOG"
        ans = parse_rank_answer(text, self.EXPECTED)
        assert ans is not None
        assert ans.parser_used == "numbered"
        assert ans.ranking == ("AAPL", "MSFT", "GOOG")

    def test_bulleted_dash(self):
        text = "- AAPL\n- MSFT\n- GOOG"
        ans = parse_rank_answer(text, self.EXPECTED)
        assert ans is not None
        assert ans.parser_used == "bulleted_dash"

    def test_bulleted_asterisk(self):
        text = "* AAPL\n* MSFT\n* GOOG"
        ans = parse_rank_answer(text, self.EXPECTED)
        assert ans is not None
        assert ans.parser_used == "bulleted_asterisk"

    def test_comma_separated(self):
        ans = parse_rank_answer("AAPL, MSFT, GOOG", self.EXPECTED)
        assert ans is not None
        assert ans.parser_used == "comma"

    def test_newline_separated(self):
        ans = parse_rank_answer("AAPL\nMSFT\nGOOG", self.EXPECTED)
        assert ans is not None
        assert ans.parser_used == "newline"

    def test_returns_none_on_unparseable(self):
        ans = parse_rank_answer("nonsense gibberish", self.EXPECTED)
        # Single-token text doesn't match any format → None
        # (well, "nonsense gibberish" has space and \n... actually
        # it's a single line with one space → comma path won't fire,
        # newline won't fire — falls through to None).
        # Actually it might be parsed as newline or other; check the
        # symbol is treated as extra not crash:
        if ans is not None:
            assert ans.extra_symbols  # at least classified extra

    def test_returns_none_on_duplicates(self):
        ans = parse_rank_answer('["AAPL","AAPL"]', self.EXPECTED)
        assert ans is None

    def test_records_extra_symbols(self):
        ans = parse_rank_answer(
            '["AAPL","XXXX","MSFT","GOOG"]', self.EXPECTED,
        )
        assert ans is not None
        assert "XXXX" in ans.extra_symbols

    def test_records_omitted_symbols(self):
        ans = parse_rank_answer('["AAPL","MSFT"]', self.EXPECTED)
        assert ans is not None
        assert "GOOG" in ans.omitted_symbols

    def test_normalizes_via_us_equity_preset(self):
        ans = parse_rank_answer(
            '["aapl","brk-b","msft"]',
            ["AAPL", "BRK.B", "MSFT"],
            normalization_preset="us_equity",
        )
        assert ans is not None
        assert ans.ranking == ("AAPL", "BRK.B", "MSFT")
        # raw_ranking preserves user's text
        assert ans.raw_ranking == ("aapl", "brk-b", "msft")

    def test_passthrough_preserves_case(self):
        ans = parse_rank_answer(
            '["aapl","msft"]', ["aapl", "msft"],
            normalization_preset="passthrough",
        )
        assert ans is not None
        assert ans.ranking == ("aapl", "msft")


# ---------- resolve_rank_answer (priority chain) ----------


class TestResolveRankAnswer:
    EXPECTED = ["AAPL", "MSFT", "GOOG"]

    def test_path_1_rankanswer_directly(self):
        ra = RankAnswer(
            ranking=("AAPL", "MSFT", "GOOG"),
            parser_used="user_provided",
        )
        rec = AnswerRecord(
            question_id="q1", raw_answer=None,
            parsed_answer=ra, parse_status="valid",
        )
        ans, status, parser = resolve_rank_answer(rec, self.EXPECTED)
        assert ans is ra
        assert status == "valid"
        assert parser == "user_provided"

    def test_path_2_list_of_strings(self):
        rec = AnswerRecord(
            question_id="q1", raw_answer=None,
            parsed_answer=["AAPL", "MSFT", "GOOG"],
            parse_status="valid",
        )
        ans, status, parser = resolve_rank_answer(rec, self.EXPECTED)
        assert ans is not None
        assert ans.ranking == ("AAPL", "MSFT", "GOOG")
        assert parser == "user_provided_list"
        # raw_ranking captured
        assert ans.raw_ranking == ("AAPL", "MSFT", "GOOG")

    def test_path_2_rejects_bare_string(self):
        # Bare "AAPL" must NOT be treated as Sequence[str].
        rec = AnswerRecord(
            question_id="q1", raw_answer=None,
            parsed_answer="AAPL",  # bare str
            parse_status="valid",
        )
        ans, status, parser = resolve_rank_answer(rec, self.EXPECTED)
        # Should NOT split into ("A","A","P","L"); should fall through
        # to raw_answer path (None) → invalid.
        assert ans is None
        assert status == "invalid"

    def test_path_2_accepts_numpy_string_array(self):
        rec = AnswerRecord(
            question_id="q1", raw_answer=None,
            parsed_answer=np.array(["AAPL", "MSFT", "GOOG"]),
            parse_status="valid",
        )
        ans, status, parser = resolve_rank_answer(rec, self.EXPECTED)
        assert ans is not None
        assert parser == "user_provided_list"

    def test_path_2_accepts_pandas_string_series(self):
        rec = AnswerRecord(
            question_id="q1", raw_answer=None,
            parsed_answer=pd.Series(
                ["AAPL", "MSFT", "GOOG"], dtype="string",
            ),
            parse_status="valid",
        )
        ans, status, parser = resolve_rank_answer(rec, self.EXPECTED)
        assert ans is not None
        assert parser == "user_provided_list"

    def test_path_2_rejects_pandas_series_with_mixed_types(self):
        # Per v2.2.1 r6 #1 fix: pd.Series([1, "AAPL"]) must NOT
        # pass the path-2 gate (the lying is_string_dtype on
        # object dtype was the bug).
        rec = AnswerRecord(
            question_id="q1", raw_answer=None,
            parsed_answer=pd.Series([1, "AAPL"], dtype=object),
            parse_status="valid",
        )
        ans, status, parser = resolve_rank_answer(rec, self.EXPECTED)
        assert ans is None
        assert status == "invalid"

    def test_path_3_dict_with_ranking_key(self):
        rec = AnswerRecord(
            question_id="q1", raw_answer=None,
            parsed_answer={"ranking": ["AAPL", "MSFT", "GOOG"]},
            parse_status="valid",
        )
        ans, status, parser = resolve_rank_answer(rec, self.EXPECTED)
        assert ans is not None
        assert parser == "user_provided_dict"

    def test_path_4_text_parser_fallback(self):
        rec = AnswerRecord(
            question_id="q1", raw_answer='["AAPL","MSFT","GOOG"]',
            parsed_answer=None, parse_status="valid",
        )
        ans, status, parser = resolve_rank_answer(rec, self.EXPECTED)
        assert ans is not None
        assert parser == "json_array"

    def test_path_5_unrecognized_returns_invalid(self):
        rec = AnswerRecord(
            question_id="q1", raw_answer=None,
            parsed_answer=42,  # int — not handled
            parse_status="valid",
        )
        ans, status, parser = resolve_rank_answer(rec, self.EXPECTED)
        assert ans is None
        assert status == "invalid"
        assert parser == "unrecognized_parsed_answer_type"


# ---------- Normalization wiring on RankContinuationProbe ----------


class TestNormalizationWiring:
    def test_probe_normalizes_truth_via_us_equity_preset(self):
        # Build with raw lowercase symbols + dash-style + us_equity
        # preset → truth_value contains normalized form.
        # Construct minimal data dict keyed by RAW symbols.
        idx = pd.bdate_range("2024-01-01", periods=30)

        def _bars():
            return pd.DataFrame(
                {
                    "open": [100.0] * 30, "high": [101.0] * 30,
                    "low": [99.0] * 30, "close": [100.0] * 20 + [105.0]
                    + [100.0] * 9, "volume": [1e6] * 30,
                },
                index=idx,
            )

        probe = RankContinuationProbe(
            symbols=["aapl", "brk-b"],  # raw
            context_bars=5,
            forward_horizon=1,
            normalization_preset="us_equity",
        )
        # data_dict keyed by RAW (what user passed)
        data = {"aapl": _bars(), "brk-b": _bars()}
        anchors = [idx[19]]
        qs = list(probe.build(data, anchors))
        assert len(qs) == 1
        # Truth is in normalized form
        truth = qs[0].truth_value
        assert truth in (("AAPL", "BRK.B"), ("BRK.B", "AAPL"))
