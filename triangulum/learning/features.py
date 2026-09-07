"""
Feature extraction.

The learning problem this engine actually needs to solve is NOT "predict the
price". It is:

    Given that a cycle *appears* to be worth +4 bps, what is the probability
    that I actually capture it, and how much will I really get?

That is a much better-posed problem than price prediction. It has a clean label
(did the cycle complete?), a fast feedback loop (seconds), abundant data (every
attempt, including the rejected ones), and a stable relationship -- market
microstructure changes far more slowly than prices do.

The features below are ordered by how much they matter in practice:

**Staleness** dominates. The single best predictor of a phantom opportunity is
the age of the oldest book in the cycle. An "edge" from a book that has not
ticked in 300ms is usually an edge that closed 299ms ago.

**Spread** and **depth ratio** come next: a cycle whose legs are wide relative
to the edge, or whose size consumes most of the touch, will not fill at the
price that made it look attractive.

**Edge magnitude** is genuinely informative but in the *opposite* direction from
intuition: an unusually large apparent edge is more likely to be a data error
than a gift. The model learns this inverted relationship on its own, which is a
good sanity check that the pipeline works.

**Venue latency** and **time of day** capture the regime: the same cycle has a
different fill probability at 03:00 UTC than at 14:30 UTC when US equities open
and crypto volatility spikes.

Features are hashed into a fixed-width vector so the model never needs a schema
migration when a new one is added -- important for a system meant to run for
months while being edited.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Mapping, Sequence

from triangulum.core.decimal_math import D, ZERO, safe_div
from triangulum.core.types import CyclePlan, Opportunity, Side
from triangulum.marketdata.book_manager import BookManager

__all__ = ["FeatureVector", "FeatureExtractor", "FEATURE_NAMES"]


FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "log_edge_bps",
    "edge_bps",
    "max_book_age_ms",
    "mean_book_age_ms",
    "log_book_age",
    "max_spread_bps",
    "mean_spread_bps",
    "spread_to_edge_ratio",
    "min_depth_ratio",
    "mean_depth_ratio",
    "max_levels_consumed",
    "cycle_length",
    "is_cross_venue",
    "venue_latency_ms",
    "venue_latency_zscore",
    "recent_fill_rate",
    "recent_attempts",
    "cycle_recency_ms",
    "hour_sin",
    "hour_cos",
    "book_imbalance_mean",
    "book_imbalance_worst",
    "volatility_zscore",
    "notional_log",
    "drag_bps",
    "fee_bps",
    "maker_legs",
    "quote_is_stable",
    "consecutive_failures",
)


# --------------------------------------------------------------------------
# Feature scaling
# --------------------------------------------------------------------------
#
# Every feature is divided by a natural scale so it lands in roughly [-3, 3].
#
# This is not cosmetic. A linear model with a single learning rate cannot cope
# with features spanning orders of magnitude: the gradient for a feature valued
# 6000 is six million times the gradient for one valued 0.001, and the update
# that is right for one is catastrophically wrong for the other. Measured on
# this engine's own feature set before scaling was added, the slippage
# regression diverged to a prediction of 6,637 bps -- a number with no physical
# meaning -- and the fill model's skill went to -13.8, far worse than always
# predicting the base rate.
#
# Adaptive per-coordinate rates (FTRL, RMSProp) help but do not rescue it: they
# adapt the *step*, not the feature's contribution to the score, so a
# 6000-valued feature still dominates the linear combination.
#
# The scales below are chosen as "a typical large value" for each feature, so
# a normal observation lands near 1 and an extreme one near 3. Clipping at the
# end bounds the influence of genuinely pathological inputs -- a book age of 40
# seconds after a disconnect should not produce a gradient 100x larger than one
# of 400ms.

FEATURE_SCALES: Mapping[str, float] = {
    "edge_bps": 10.0,
    "log_edge_bps": 2.5,
    "max_book_age_ms": 250.0,
    "mean_book_age_ms": 250.0,
    "log_book_age": 6.0,
    "max_spread_bps": 20.0,
    "mean_spread_bps": 20.0,
    "spread_to_edge_ratio": 10.0,
    "min_depth_ratio": 5.0,
    "mean_depth_ratio": 5.0,
    "max_levels_consumed": 5.0,
    "cycle_length": 4.0,
    "venue_latency_ms": 100.0,
    "venue_latency_zscore": 3.0,
    "recent_attempts": 5.0,
    "cycle_recency_ms": 15.0,
    "volatility_zscore": 3.0,
    "notional_log": 10.0,
    "drag_bps": 20.0,
    "fee_bps": 40.0,
    "maker_legs": 3.0,
    "consecutive_failures": 10.0,
    "book_imbalance_mean": 1.0,
    "book_imbalance_worst": 1.0,
    "recent_fill_rate": 1.0,
    # bias, hour_sin, hour_cos, is_cross_venue, quote_is_stable are already
    # in [-1, 1] and are left alone.
}

FEATURE_CLIP = 3.0


def scale_features(values: Mapping[str, float]) -> dict[str, float]:
    """Divide by natural scale, then clip. See the note above."""
    out: dict[str, float] = {}
    for name, value in values.items():
        scaled = value / FEATURE_SCALES.get(name, 1.0)
        out[name] = max(-FEATURE_CLIP, min(FEATURE_CLIP, scaled))
    return out


@dataclass(slots=True)
class FeatureVector:
    """Dense named features plus their hashed sparse representation."""

    values: dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float = 0.0) -> float:
        return self.values.get(name, default)

    def to_dense(self, names: Sequence[str] = FEATURE_NAMES) -> list[float]:
        return [self.values.get(n, 0.0) for n in names]

    def to_hashed(self, bits: int = 18) -> dict[int, float]:
        """
        Hash features into ``2**bits`` buckets.

        Hashing means a new feature can be added without retraining from
        scratch or migrating a stored schema -- it simply lands in a new bucket
        whose weight starts at zero. Collisions at 18 bits (262,144 buckets)
        with ~30 features are effectively impossible.
        """
        size = 1 << bits
        out: dict[int, float] = {}
        for name, value in self.values.items():
            if value == 0.0:
                continue
            index = _stable_hash(name) % size
            out[index] = out.get(index, 0.0) + value
        return out

    def __repr__(self) -> str:  # pragma: no cover
        top = sorted(self.values.items(), key=lambda kv: -abs(kv[1]))[:5]
        return "FeatureVector(" + ", ".join(f"{k}={v:.3f}" for k, v in top) + ")"


def _stable_hash(text: str) -> int:
    """
    FNV-1a. Python's ``hash`` is randomised per process, which would make a
    persisted model meaningless after a restart.
    """
    h = 2166136261
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


class FeatureExtractor:
    """Builds feature vectors from opportunities and plans."""

    def __init__(
        self,
        books: BookManager,
        *,
        stablecoins: frozenset[str] = frozenset({"USDT", "USDC", "BUSD", "DAI", "FDUSD"}),
    ) -> None:
        self.books = books
        self.stablecoins = stablecoins
        self._venue_latency_ms: dict[str, float] = {}
        self._venue_latency_std: dict[str, float] = {}
        self._cycle_attempts: dict[str, int] = {}
        self._cycle_fills: dict[str, int] = {}
        self._cycle_last_ns: dict[str, int] = {}
        self._consecutive_failures = 0
        self._volatility_z: dict[str, float] = {}

    # -- state updates -----------------------------------------------------

    def observe_latency(self, venue: str, mean_ms: float, std_ms: float = 0.0) -> None:
        self._venue_latency_ms[venue] = mean_ms
        self._venue_latency_std[venue] = std_ms

    def observe_volatility(self, symbol_key: str, zscore: float) -> None:
        self._volatility_z[symbol_key] = zscore

    def observe_outcome(self, cycle_key: str, filled: bool, now_ns: int) -> None:
        self._cycle_attempts[cycle_key] = self._cycle_attempts.get(cycle_key, 0) + 1
        if filled:
            self._cycle_fills[cycle_key] = self._cycle_fills.get(cycle_key, 0) + 1
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
        self._cycle_last_ns[cycle_key] = now_ns

    # -- extraction --------------------------------------------------------

    def extract(
        self,
        opportunity: Opportunity,
        *,
        now_ns: int,
        plan: CyclePlan | None = None,
        notional: Decimal = ZERO,
    ) -> FeatureVector:
        values: dict[str, float] = {"bias": 1.0}
        legs = opportunity.legs
        cycle_key = opportunity.path

        edge = float(opportunity.gross_edge_bps)
        values["edge_bps"] = edge
        # Log-transform: the relationship between edge and fill probability is
        # strongly non-linear, and an untransformed 400 bps outlier would
        # otherwise dominate every gradient step.
        values["log_edge_bps"] = math.log1p(max(0.0, edge))

        # -- book state per leg --
        spreads: list[float] = []
        ages: list[float] = []
        depth_ratios: list[float] = []
        imbalances: list[float] = []
        levels: list[int] = []

        # Book ages recorded AT DETECTION are what matter. Recomputing them here
        # measures the age at *evaluation* time, which is a different and much
        # smaller number -- and it silently collapses to zero whenever the
        # detector and the gate run in the same tick, destroying the single most
        # predictive feature in the set.
        recorded_ages = opportunity.book_ages_ns

        for index, leg in enumerate(legs):
            book = self.books.get(leg.symbol)
            recorded = (
                recorded_ages[index] / 1e6
                if index < len(recorded_ages) else None
            )

            if book is None or not book.initialized:
                spreads.append(100.0)
                ages.append(recorded if recorded is not None else 1000.0)
                depth_ratios.append(0.0)
                imbalances.append(0.0)
                continue

            spreads.append(float(book.spread_bps))
            ages.append(
                recorded if recorded is not None else book.age_ns(now_ns) / 1e6
            )
            quote = book.quote()
            imbalances.append(float(quote.imbalance))

            touch_size = (
                book.asks.best_size() if leg.side is Side.BUY else book.bids.best_size()
            )
            touch_price = book.best_ask if leg.side is Side.BUY else book.best_bid
            touch_notional = touch_size * touch_price
            want = notional if notional > 0 else opportunity.reference_notional
            # >1 means the touch alone covers our order; <1 means we walk.
            depth_ratios.append(float(safe_div(touch_notional, want)) if want > 0 else 0.0)

            if plan is not None and index < len(plan.legs):
                levels.append(plan.legs[index].levels_consumed)

        values["max_spread_bps"] = max(spreads) if spreads else 0.0
        values["mean_spread_bps"] = sum(spreads) / len(spreads) if spreads else 0.0
        values["max_book_age_ms"] = max(ages) if ages else 0.0
        values["mean_book_age_ms"] = sum(ages) / len(ages) if ages else 0.0
        values["log_book_age"] = math.log1p(max(0.0, max(ages) if ages else 0.0))
        values["min_depth_ratio"] = min(depth_ratios) if depth_ratios else 0.0
        values["mean_depth_ratio"] = (
            sum(depth_ratios) / len(depth_ratios) if depth_ratios else 0.0
        )
        values["book_imbalance_mean"] = (
            sum(imbalances) / len(imbalances) if imbalances else 0.0
        )
        values["book_imbalance_worst"] = (
            min(imbalances, key=abs) if imbalances else 0.0
        )
        values["max_levels_consumed"] = float(max(levels)) if levels else 1.0

        # The ratio that matters most: an edge of 3 bps behind a 12 bps spread
        # is not an edge, it is a rounding artifact of the mid price.
        total_spread = sum(spreads)
        values["spread_to_edge_ratio"] = (
            total_spread / edge if edge > 0.01 else 100.0
        )

        # -- structure --
        values["cycle_length"] = float(len(legs))
        values["is_cross_venue"] = 1.0 if len(set(opportunity.venues)) > 1 else 0.0
        values["quote_is_stable"] = (
            1.0 if opportunity.start_asset.code in self.stablecoins else 0.0
        )

        # -- venue --
        venue = opportunity.venues[0] if opportunity.venues else ""
        latency = self._venue_latency_ms.get(venue, 50.0)
        values["venue_latency_ms"] = latency
        std = self._venue_latency_std.get(venue, 0.0)
        values["venue_latency_zscore"] = (
            (latency - 50.0) / std if std > 1e-6 else 0.0
        )

        # -- history --
        attempts = self._cycle_attempts.get(cycle_key, 0)
        fills = self._cycle_fills.get(cycle_key, 0)
        values["recent_attempts"] = math.log1p(attempts)
        values["recent_fill_rate"] = fills / attempts if attempts else 0.5
        last = self._cycle_last_ns.get(cycle_key)
        values["cycle_recency_ms"] = (
            math.log1p((now_ns - last) / 1e6) if last else 15.0
        )
        values["consecutive_failures"] = float(min(20, self._consecutive_failures))

        # -- time of day, as a circular pair so 23:59 is adjacent to 00:01 --
        hour = datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc).hour
        values["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        values["hour_cos"] = math.cos(2 * math.pi * hour / 24)

        # -- volatility regime --
        zs = [
            self._volatility_z.get(leg.symbol.key, 0.0) for leg in legs
        ]
        values["volatility_zscore"] = max(zs, key=abs) if zs else 0.0

        # -- plan-derived --
        if plan is not None:
            values["drag_bps"] = float(plan.slippage_bps)
            values["fee_bps"] = float(plan.fee_bps)
            values["maker_legs"] = float(sum(
                1 for lp in plan.legs if lp.order_type.value == "post_only"
            ))
            values["notional_log"] = math.log1p(float(plan.start_amount))
        else:
            values["notional_log"] = math.log1p(float(notional or opportunity.reference_notional))

        # Scale and clip before returning. Everything downstream -- the FTRL
        # model, the ridge, the hashing -- assumes bounded inputs.
        return FeatureVector(values=scale_features(values))

    def stats(self) -> dict[str, object]:
        return {
            "tracked_cycles": len(self._cycle_attempts),
            "total_attempts": sum(self._cycle_attempts.values()),
            "total_fills": sum(self._cycle_fills.values()),
            "consecutive_failures": self._consecutive_failures,
            "venues_with_latency": len(self._venue_latency_ms),
        }
