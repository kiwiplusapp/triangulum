"""
Domain model.

Everything downstream speaks these types. They are frozen dataclasses with
``slots=True`` because the hot path constructs tens of thousands of them per
second and attribute dictionaries are pure overhead at that rate.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Iterator, Mapping, Sequence

from triangulum.core.decimal_math import D, ONE, ZERO, bps, safe_div

__all__ = [
    "Side",
    "OrderType",
    "TimeInForce",
    "OrderStatus",
    "Liquidity",
    "AssetClass",
    "ExecutionMode",
    "RunMode",
    "Asset",
    "Symbol",
    "PriceLevel",
    "BookSnapshot",
    "Quote",
    "Trade",
    "Balance",
    "Order",
    "Fill",
    "Leg",
    "LegPlan",
    "CyclePlan",
    "Opportunity",
    "ExecutionResult",
    "CycleOutcome",
    "new_id",
]


def new_id(prefix: str = "") -> str:
    """Short, sortable-enough correlation id. Not a UUID on the wire."""
    raw = uuid.uuid4().hex[:16]
    return f"{prefix}{raw}" if prefix else raw


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class Side(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        """+1 when the trade increases base inventory, -1 when it decreases."""
        return 1 if self is Side.BUY else -1


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"   # limit that cancels rather than crossing
    STOP_MARKET = "stop_market"


class TimeInForce(str, enum.Enum):
    GTC = "gtc"
    IOC = "ioc"   # take what is there, cancel the rest
    FOK = "fok"   # all or nothing -- the arbitrageur's friend
    GTX = "gtx"   # post-only / add-liquidity-only


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        return self in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def any_fill(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)


class Liquidity(str, enum.Enum):
    MAKER = "maker"
    TAKER = "taker"
    UNKNOWN = "unknown"


class AssetClass(str, enum.Enum):
    CRYPTO = "crypto"
    FIAT = "fiat"
    EQUITY = "equity"
    FX = "fx"
    METAL = "metal"
    STABLECOIN = "stablecoin"


class ExecutionMode(str, enum.Enum):
    """
    How a cycle's legs are worked. The choice dominates both the fee bill and
    the fill risk, which is why the bandit treats it as an arm.
    """

    TTT = "taker_taker_taker"      # all IOC/FOK. Fast, certain, expensive.
    MTT = "maker_taker_taker"      # post-only leg 1, then sweep. Cheap, risky.
    TMT = "taker_maker_taker"
    ADAPTIVE = "adaptive"          # let the bandit pick per-opportunity


class RunMode(str, enum.Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

    @property
    def touches_real_money(self) -> bool:
        return self is RunMode.LIVE


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Asset:
    """A unit of account. ``USDT``, ``BTC``, ``EUR``, ``AAPL``."""

    code: str
    asset_class: AssetClass = AssetClass.CRYPTO

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", self.code.upper())

    def __str__(self) -> str:
        return self.code

    @property
    def is_cash_like(self) -> bool:
        return self.asset_class in (AssetClass.FIAT, AssetClass.STABLECOIN)


@dataclass(frozen=True, slots=True)
class Symbol:
    """
    A tradable pair on a specific venue.

    ``base``/``quote`` are semantic: a BUY of BTC/USDT spends USDT and receives
    BTC. ``venue_symbol`` is the wire format, which differs per venue for the
    same economic pair (``BTCUSDT`` on Binance, ``XBT/USDT`` on Kraken,
    ``BTC-USDT`` on Coinbase). Normalizing this away is the market-data layer's
    job; nothing above it should ever see a venue string.
    """

    base: Asset
    quote: Asset
    venue: str
    venue_symbol: str = ""

    def __post_init__(self) -> None:
        if not self.venue_symbol:
            object.__setattr__(self, "venue_symbol", f"{self.base}{self.quote}")

    @property
    def canonical(self) -> str:
        return f"{self.base}/{self.quote}"

    @property
    def key(self) -> str:
        return f"{self.venue}:{self.canonical}"

    def __str__(self) -> str:
        return self.key

    def other(self, asset: Asset) -> Asset:
        """Given one side of the pair, return the other."""
        if asset == self.base:
            return self.quote
        if asset == self.quote:
            return self.base
        raise ValueError(f"{asset} is not part of {self.canonical}")

    def side_to_convert(self, frm: Asset, to: Asset) -> Side:
        """
        Which side converts ``frm`` into ``to`` on this pair?

        Spending quote to get base is a BUY; spending base to get quote is a
        SELL. This one-liner is the hinge the whole graph layer turns on.
        """
        if frm == self.quote and to == self.base:
            return Side.BUY
        if frm == self.base and to == self.quote:
            return Side.SELL
        raise ValueError(f"cannot convert {frm}->{to} on {self.canonical}")


# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PriceLevel:
    price: Decimal
    size: Decimal

    @property
    def notional(self) -> Decimal:
        return self.price * self.size


@dataclass(frozen=True, slots=True)
class Quote:
    """Top of book. The cheap representation used for edge screening."""

    symbol: Symbol
    bid: Decimal
    ask: Decimal
    bid_size: Decimal = ZERO
    ask_size: Decimal = ZERO
    ts_venue_ns: int = 0
    ts_local_ns: int = 0

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / D(2)

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> Decimal:
        m = self.mid
        return bps(safe_div(self.spread, m)) if m > 0 else ZERO

    @property
    def microprice(self) -> Decimal:
        """
        Size-weighted mid. A better fair-value estimate than the plain mid when
        the book is lopsided, and the anchor the slippage model regresses on.
        """
        total = self.bid_size + self.ask_size
        if total <= 0:
            return self.mid
        return (self.bid * self.ask_size + self.ask * self.bid_size) / total

    @property
    def imbalance(self) -> Decimal:
        """(bid_size - ask_size) / (bid_size + ask_size), in [-1, 1]."""
        total = self.bid_size + self.ask_size
        if total <= 0:
            return ZERO
        return (self.bid_size - self.ask_size) / total

    @property
    def crossed(self) -> bool:
        return self.bid >= self.ask > 0

    def age_ns(self, now_ns: int) -> int:
        return max(0, now_ns - self.ts_local_ns)


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """
    Immutable point-in-time L2 book, deepest-first-sorted.

    ``bids`` descend by price, ``asks`` ascend. The invariant is enforced at
    construction in the mutable book and assumed everywhere here.
    """

    symbol: Symbol
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]
    sequence: int = 0
    ts_venue_ns: int = 0
    ts_local_ns: int = 0

    @property
    def best_bid(self) -> Decimal:
        return self.bids[0].price if self.bids else ZERO

    @property
    def best_ask(self) -> Decimal:
        return self.asks[0].price if self.asks else ZERO

    @property
    def quote(self) -> Quote:
        return Quote(
            symbol=self.symbol,
            bid=self.best_bid,
            ask=self.best_ask,
            bid_size=self.bids[0].size if self.bids else ZERO,
            ask_size=self.asks[0].size if self.asks else ZERO,
            ts_venue_ns=self.ts_venue_ns,
            ts_local_ns=self.ts_local_ns,
        )

    def side(self, side: Side) -> tuple[PriceLevel, ...]:
        """Levels you consume when taking liquidity on ``side``."""
        return self.asks if side is Side.BUY else self.bids

    def depth_notional(self, side: Side, levels: int = 10) -> Decimal:
        return sum((lvl.notional for lvl in self.side(side)[:levels]), ZERO)

    def __bool__(self) -> bool:
        return bool(self.bids and self.asks)


@dataclass(frozen=True, slots=True)
class Trade:
    """A public trade print."""

    symbol: Symbol
    price: Decimal
    size: Decimal
    side: Side
    ts_venue_ns: int = 0
    ts_local_ns: int = 0
    trade_id: str = ""


# --------------------------------------------------------------------------
# Account
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Balance:
    asset: Asset
    free: Decimal
    locked: Decimal = ZERO

    @property
    def total(self) -> Decimal:
        return self.free + self.locked


# --------------------------------------------------------------------------
# Orders and fills
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Order:
    symbol: Symbol
    side: Side
    quantity: Decimal
    order_type: OrderType = OrderType.LIMIT
    price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.IOC
    client_order_id: str = field(default_factory=lambda: new_id("tri-"))
    venue_order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = ZERO
    average_price: Decimal = ZERO
    fee_paid: Decimal = ZERO
    fee_asset: Asset | None = None
    reduce_only: bool = False
    ts_created_ns: int = 0
    ts_submitted_ns: int = 0
    ts_final_ns: int = 0
    tag: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def remaining(self) -> Decimal:
        return max(ZERO, self.quantity - self.filled_quantity)

    @property
    def fill_ratio(self) -> Decimal:
        return safe_div(self.filled_quantity, self.quantity)

    @property
    def notional(self) -> Decimal:
        px = self.average_price if self.average_price > 0 else (self.price or ZERO)
        return self.filled_quantity * px

    @property
    def latency_ns(self) -> int:
        if self.ts_final_ns and self.ts_submitted_ns:
            return self.ts_final_ns - self.ts_submitted_ns
        return 0

    def with_status(self, status: OrderStatus, **changes: Any) -> "Order":
        return replace(self, status=status, **changes)


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    symbol: Symbol
    side: Side
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_asset: Asset
    liquidity: Liquidity = Liquidity.TAKER
    ts_ns: int = 0
    trade_id: str = field(default_factory=lambda: new_id("f-"))

    @property
    def notional(self) -> Decimal:
        return self.price * self.quantity

    @property
    def fee_bps(self) -> Decimal:
        return bps(safe_div(self.fee, self.notional))


# --------------------------------------------------------------------------
# Cycle planning
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Leg:
    """One conversion hop: spend ``from_asset``, receive ``to_asset``."""

    symbol: Symbol
    side: Side
    from_asset: Asset
    to_asset: Asset

    @property
    def venue(self) -> str:
        return self.symbol.venue

    def __str__(self) -> str:
        return f"{self.from_asset}->{self.to_asset} via {self.symbol.canonical} ({self.side.value})"


@dataclass(frozen=True, slots=True)
class LegPlan:
    """A leg with its sized, priced, fee-adjusted execution parameters."""

    leg: Leg
    input_amount: Decimal          # units of from_asset committed
    expected_output: Decimal       # units of to_asset expected, net of fees
    quantity: Decimal              # order quantity, in base units, lot-aligned
    limit_price: Decimal
    order_type: OrderType
    time_in_force: TimeInForce
    expected_fee: Decimal
    expected_fee_asset: Asset
    expected_slippage_bps: Decimal = ZERO
    levels_consumed: int = 1
    book_sequence: int = 0

    @property
    def effective_rate(self) -> Decimal:
        """Units of ``to_asset`` per unit of ``from_asset``, net of everything."""
        return safe_div(self.expected_output, self.input_amount)

    @property
    def notional_quote(self) -> Decimal:
        return self.quantity * self.limit_price


@dataclass(frozen=True, slots=True)
class CyclePlan:
    """A fully sized, executable arbitrage cycle."""

    cycle_id: str
    legs: tuple[LegPlan, ...]
    start_asset: Asset
    start_amount: Decimal
    expected_end_amount: Decimal
    execution_mode: ExecutionMode
    gross_edge_bps: Decimal
    fee_bps: Decimal
    slippage_bps: Decimal
    net_edge_bps: Decimal
    fill_probability: Decimal = ONE
    expected_value_bps: Decimal = ZERO
    unwind_cost_bps: Decimal = ZERO
    ts_created_ns: int = 0
    features: Mapping[str, float] = field(default_factory=dict)
    bandit_arm: str = ""

    def __iter__(self) -> Iterator[LegPlan]:
        return iter(self.legs)

    def __len__(self) -> int:
        return len(self.legs)

    @property
    def venues(self) -> tuple[str, ...]:
        seen: list[str] = []
        for lp in self.legs:
            if lp.leg.venue not in seen:
                seen.append(lp.leg.venue)
        return tuple(seen)

    @property
    def is_cross_venue(self) -> bool:
        return len(self.venues) > 1

    @property
    def path(self) -> str:
        assets = [str(self.start_asset)] + [str(lp.leg.to_asset) for lp in self.legs]
        return " -> ".join(assets)

    @property
    def expected_profit(self) -> Decimal:
        return self.expected_end_amount - self.start_amount

    def validate(self) -> None:
        """Structural sanity. Cheap, and it has caught real planner bugs."""
        if not self.legs:
            raise ValueError("cycle has no legs")
        cursor = self.start_asset
        for i, lp in enumerate(self.legs):
            if lp.leg.from_asset != cursor:
                raise ValueError(
                    f"leg {i} starts at {lp.leg.from_asset}, expected {cursor}"
                )
            cursor = lp.leg.to_asset
        if cursor != self.start_asset:
            raise ValueError(f"cycle does not close: ends at {cursor}, started at {self.start_asset}")


@dataclass(frozen=True, slots=True)
class Opportunity:
    """
    A detected cycle before sizing. The graph layer emits these; the planner
    turns the promising ones into :class:`CyclePlan`.
    """

    opportunity_id: str
    legs: tuple[Leg, ...]
    start_asset: Asset
    gross_edge_bps: Decimal
    reference_notional: Decimal
    ts_detected_ns: int
    book_ages_ns: tuple[int, ...] = ()
    venues: tuple[str, ...] = ()

    @property
    def max_book_age_ns(self) -> int:
        """
        Staleness of the *oldest* book in the cycle.

        This single feature separates real opportunities from phantoms better
        than any other: an "edge" computed from a book that has not ticked in
        400ms is usually an edge that closed 399ms ago.
        """
        return max(self.book_ages_ns) if self.book_ages_ns else 0

    @property
    def path(self) -> str:
        assets = [str(self.start_asset)] + [str(l.to_asset) for l in self.legs]
        return " -> ".join(assets)


# --------------------------------------------------------------------------
# Execution results
# --------------------------------------------------------------------------


class CycleOutcome(str, enum.Enum):
    COMPLETED = "completed"            # all legs filled, cycle closed
    ABORTED_PRE_TRADE = "aborted"      # rejected before any capital moved
    PARTIAL_UNWOUND = "partial_unwound"    # failed mid-cycle, inventory flattened
    PARTIAL_STUCK = "partial_stuck"        # failed mid-cycle, inventory held
    REJECTED = "rejected"
    ERROR = "error"

    @property
    def committed_capital(self) -> bool:
        return self not in (CycleOutcome.ABORTED_PRE_TRADE, CycleOutcome.REJECTED)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """What actually happened, for the ledger and for the learner's label."""

    cycle_id: str
    outcome: CycleOutcome
    plan: CyclePlan
    fills: tuple[Fill, ...] = ()
    realized_start_amount: Decimal = ZERO
    realized_end_amount: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    realized_pnl_bps: Decimal = ZERO
    total_fees: Decimal = ZERO
    legs_filled: int = 0
    latency_ns: int = 0
    error: str = ""
    ts_start_ns: int = 0
    ts_end_ns: int = 0
    residual_inventory: Mapping[str, Decimal] = field(default_factory=dict)

    @property
    def fully_filled(self) -> bool:
        return self.legs_filled == len(self.plan.legs)

    @property
    def label_filled(self) -> int:
        """Binary label for the fill-probability model."""
        return 1 if self.outcome is CycleOutcome.COMPLETED else 0

    @property
    def slippage_bps(self) -> Decimal:
        """Realized minus expected. The regression target of the slippage model."""
        return self.plan.net_edge_bps - self.realized_pnl_bps

    def summary(self) -> str:
        return (
            f"[{self.outcome.value}] {self.plan.path} "
            f"exp={self.plan.net_edge_bps:.2f}bps "
            f"real={self.realized_pnl_bps:.2f}bps "
            f"legs={self.legs_filled}/{len(self.plan.legs)} "
            f"lat={self.latency_ns / 1e6:.1f}ms"
        )
