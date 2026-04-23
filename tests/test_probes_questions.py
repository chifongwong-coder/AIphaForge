"""v2.0 M3 — tests for question templates, sampling, and exporter."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes import (
    DEFAULT_TEMPLATES,
    BarRangePct,
    CloseQuestion,
    CloseVsOpen,
    GapVsPrevClose,
    HighQuestion,
    LowQuestion,
    OpenQuestion,
    QuestionPromptRecord,
    ReturnSign,
    build_question_set,
    build_question_sets_multi,
    sample_dates,
)


def _ohlcv(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
    opens = closes * (1.0 + rng.normal(0.0, 0.003, size=n))
    spreads = np.abs(rng.normal(0.0, 0.005, size=n)) * closes
    highs = np.maximum(opens, closes) + spreads
    lows = np.minimum(opens, closes) - spreads
    vol = rng.integers(1_000_000, 10_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol},
        index=pd.bdate_range("2024-01-01", periods=n),
    )


# ---------- Same-bar OHLC numeric templates ----------

class TestOHLCTemplates:
    @pytest.mark.parametrize(
        "Template,col",
        [(OpenQuestion, "open"), (HighQuestion, "high"),
         (LowQuestion, "low"), (CloseQuestion, "close")],
    )
    def test_truth_matches_dataset(self, Template, col):
        data = _ohlcv()
        ts = data.index[10]
        q = Template().build(data, "AAPL", ts)
        assert q.answer_type == "numeric"
        assert q.truth_value == pytest.approx(data.loc[ts, col])
        assert q.tolerance is not None
        assert q.tolerance.max_range_width is not None  # built-in default cap

    def test_unknown_timestamp_raises(self):
        data = _ohlcv()
        with pytest.raises(KeyError):
            CloseQuestion().build(data, "AAPL", pd.Timestamp("2099-01-01"))


# ---------- Same-bar direction template ----------

class TestCloseVsOpen:
    def test_up_when_close_above_open(self):
        data = _ohlcv()
        ts = next(t for t in data.index if data.loc[t, "close"] > data.loc[t, "open"])
        q = CloseVsOpen().build(data, "AAPL", ts)
        assert q.answer_type == "choice"
        assert q.choices == ["up", "down", "unchanged"]
        assert q.truth_value == "up"

    def test_down_when_close_below_open(self):
        data = _ohlcv()
        ts = next(t for t in data.index if data.loc[t, "close"] < data.loc[t, "open"])
        q = CloseVsOpen().build(data, "AAPL", ts)
        assert q.truth_value == "down"

    def test_unchanged_when_within_epsilon(self):
        data = _ohlcv()
        ts = data.index[5]
        # Force open == close.
        data.loc[ts, "open"] = data.loc[ts, "close"]
        q = CloseVsOpen(sign_epsilon=1e-3).build(data, "AAPL", ts)
        assert q.truth_value == "unchanged"


# ---------- Lookback templates ----------

class TestGapVsPrevClose:
    def test_truth_is_gap_fraction(self):
        data = _ohlcv()
        ts = data.index[10]
        prev_close = float(data.iloc[9]["close"])
        bar_open = float(data.loc[ts, "open"])
        expected = (bar_open - prev_close) / prev_close
        q = GapVsPrevClose().build(data, "AAPL", ts)
        assert q.truth_value == pytest.approx(expected)

    def test_first_bar_raises(self):
        data = _ohlcv()
        with pytest.raises(ValueError, match="first bar"):
            GapVsPrevClose().build(data, "AAPL", data.index[0])


class TestBarRangePct:
    def test_truth_is_range_over_close(self):
        data = _ohlcv()
        ts = data.index[15]
        h = float(data.loc[ts, "high"])
        low = float(data.loc[ts, "low"])
        c = float(data.loc[ts, "close"])
        q = BarRangePct().build(data, "AAPL", ts)
        assert q.truth_value == pytest.approx((h - low) / c)
        assert q.tolerance is not None


class TestReturnSign:
    def test_up_when_close_above_prev_close(self):
        data = _ohlcv()
        # Find a bar where return is clearly positive.
        idx = next(
            i for i in range(1, len(data))
            if data.iloc[i]["close"] > data.iloc[i - 1]["close"] * 1.005
        )
        q = ReturnSign().build(data, "AAPL", data.index[idx])
        assert q.truth_value == "up"

    def test_unchanged_within_epsilon(self):
        data = _ohlcv()
        ts = data.index[5]
        data.loc[ts, "close"] = float(data.iloc[4]["close"])
        q = ReturnSign(sign_epsilon=1e-3).build(data, "AAPL", ts)
        assert q.truth_value == "unchanged"

    def test_first_bar_raises(self):
        data = _ohlcv()
        with pytest.raises(ValueError, match="first bar"):
            ReturnSign().build(data, "AAPL", data.index[0])


# ---------- Sampling ----------

class TestSampleDates:
    def test_returns_n_distinct_sorted(self):
        data = _ohlcv(n=100)
        sampled = sample_dates(data, n=20, seed=42)
        assert len(sampled) == 20
        assert len(set(sampled)) == 20
        assert sampled == sorted(sampled)

    def test_deterministic_per_seed(self):
        data = _ohlcv(n=100)
        a = sample_dates(data, n=10, seed=42)
        b = sample_dates(data, n=10, seed=42)
        assert a == b

    def test_excludes_warmup(self):
        data = _ohlcv(n=100)
        sampled = sample_dates(data, n=10, seed=0, start=20)
        assert all(ts >= data.index[20] for ts in sampled)

    def test_n_too_large_raises(self):
        data = _ohlcv(n=10)
        with pytest.raises(ValueError, match="only"):
            sample_dates(data, n=20, seed=0)

    def test_n_zero_raises(self):
        data = _ohlcv(n=10)
        with pytest.raises(ValueError):
            sample_dates(data, n=0, seed=0)


# ---------- Builders + dedup ----------

class TestBuildQuestionSet:
    def test_builds_across_templates_and_dates(self):
        data = _ohlcv(n=60)
        ts_list = sample_dates(data, n=5, seed=0, start=1)
        templates = [OpenQuestion(), CloseQuestion(), CloseVsOpen()]
        qs = build_question_set(data, "AAPL", ts_list, templates)
        assert len(qs) == 5 * 3

    def test_skips_lookback_unavailable(self):
        # GapVsPrevClose can't build for the first bar; should be
        # skipped silently while other templates still produce questions.
        data = _ohlcv(n=60)
        templates = [OpenQuestion(), GapVsPrevClose()]
        ts_list = [data.index[0], data.index[5]]
        qs = build_question_set(data, "AAPL", ts_list, templates)
        # First bar contributes only OpenQuestion (1 question);
        # second bar contributes both (2 questions). Total = 3.
        assert len(qs) == 3

    def test_dedup_drops_identical_questions(self):
        data = _ohlcv(n=60)
        ts = data.index[10]
        # Two identical OpenQuestion instances → same template_id +
        # symbol + timestamp + truth → deduplicated.
        qs = build_question_set(data, "AAPL", [ts, ts], [OpenQuestion()])
        assert len(qs) == 1

    def test_intraday_question_ids_distinct(self):
        # Regression: question_id must encode the full timestamp, not
        # just the date, so two intraday bars on the same calendar day
        # produce distinct ids and the answer-file lookup works.
        idx = pd.DatetimeIndex([
            "2024-03-12 09:30:00",
            "2024-03-12 14:00:00",
        ])
        intraday = pd.DataFrame(
            {
                "open": [100.0, 101.0], "high": [101.0, 102.0],
                "low": [99.0, 100.0], "close": [100.5, 101.5],
                "volume": [1e6, 1e6],
            },
            index=idx,
        )
        qs = build_question_set(intraday, "X", list(idx), [OpenQuestion()])
        assert len(qs) == 2
        ids = [q.question_id for q in qs]
        assert ids[0] != ids[1]

    def test_pipe_in_symbol_rejected(self):
        data = _ohlcv(n=60)
        with pytest.raises(ValueError, match="must not contain"):
            OpenQuestion().build(data, "BAD|SYM", data.index[5])

    def test_default_templates_cover_eight(self):
        # Plan §1.2 specifies 4 OHLC + 4 direction/relative templates.
        assert len(DEFAULT_TEMPLATES) == 8


class TestBuildQuestionSetsMulti:
    def test_multi_symbol_concatenation(self):
        data_a = _ohlcv(n=60, seed=1)
        data_b = _ohlcv(n=60, seed=2)
        ts_a = sample_dates(data_a, n=3, seed=0, start=1)
        ts_b = sample_dates(data_b, n=4, seed=0, start=1)
        qs = build_question_sets_multi(
            {"AAA": data_a, "BBB": data_b},
            {"AAA": ts_a, "BBB": ts_b},
            [OpenQuestion(), CloseQuestion()],
        )
        # 3 * 2 + 4 * 2 = 14 questions, all distinct (different symbols).
        assert len(qs) == 14
        symbols = {q.symbol for q in qs.questions}
        assert symbols == {"AAA", "BBB"}


# ---------- Exporter ----------

class TestExportSchemaSeparation:
    def test_questions_jsonl_has_no_truth_or_tolerance(self, tmp_path):
        data = _ohlcv(n=60)
        qs = build_question_set(
            data, "AAPL", [data.index[10], data.index[20]],
            [OpenQuestion(), CloseVsOpen()],
        )
        out = tmp_path / "questions.jsonl"
        qs.export_questions(str(out))
        for line in out.read_text().splitlines():
            obj = json.loads(line)
            # The structural separation is the load-bearing guarantee
            # against accidental answer-key leakage when the user
            # feeds `questions.jsonl` to the LLM.
            assert "truth_value" not in obj
            assert "tolerance" not in obj
            assert "prompt_text" in obj
            assert "answer_type" in obj

    def test_answer_key_jsonl_has_truth_and_no_prompt(self, tmp_path):
        data = _ohlcv(n=60)
        qs = build_question_set(
            data, "AAPL", [data.index[10]],
            [OpenQuestion()],
        )
        out = tmp_path / "answer_key.jsonl"
        qs.export_answer_key(str(out))
        line = out.read_text().splitlines()[0]
        obj = json.loads(line)
        assert "truth_value" in obj
        assert "tolerance" in obj
        assert "prompt_text" not in obj  # never goes to the answer key

    def test_round_trip_question_count(self, tmp_path):
        data = _ohlcv(n=60)
        qs = build_question_set(
            data, "AAPL",
            sample_dates(data, n=8, seed=0, start=1),
            list(t() for t in DEFAULT_TEMPLATES),
        )
        n_expected = len(qs)
        questions_path = tmp_path / "q.jsonl"
        key_path = tmp_path / "k.jsonl"
        qs.export_questions(str(questions_path))
        qs.export_answer_key(str(key_path))
        assert len(questions_path.read_text().splitlines()) == n_expected
        assert len(key_path.read_text().splitlines()) == n_expected


class TestQuestionSetIteration:
    def test_iter_and_len(self):
        data = _ohlcv(n=60)
        qs = build_question_set(
            data, "AAPL", [data.index[10], data.index[20]],
            [OpenQuestion()],
        )
        assert len(qs) == 2
        ids = [q.question_id for q in qs]
        assert len(ids) == 2

    def test_prompt_records_drop_truth(self):
        data = _ohlcv(n=60)
        qs = build_question_set(data, "AAPL", [data.index[10]], [OpenQuestion()])
        records = qs.prompt_records()
        assert len(records) == 1
        # The dataclass shape itself is what enforces non-leakage.
        assert isinstance(records[0], QuestionPromptRecord)
        assert "truth_value" not in QuestionPromptRecord.__dataclass_fields__
