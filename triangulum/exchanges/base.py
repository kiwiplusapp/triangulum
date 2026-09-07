"""
Exchange adapter interface.

An adapter does four things: describe the venue's instruments and rules, stream
market data, submit and track orders, and report balances. Everything above this
line is venue-agnostic, which is what makes adding a venue a contained job
rather than a refactor.

Contract notes that matter for correctness:

*``submit`` is not "fire and forget".* It returns only once the order has
reached a state the caller can act on: filled, rejected, or resting. An
arbitrage executor that submits leg 2 before knowing leg 1's outcome is not
running an arbitrage, it is running two uncorrelated directional trades.

*Idempotency.* Every order carries a client order id. On a timeout the adapter
must be able to answer "did that order actually land?" by querying that id
rather than blindly resubmitting -- double-submitting leg 1 of a cycle is a
uniquely expensive bug.

*Fees are returned, not assumed.* The venue tells us what it charged. The fee
engine's estimate is used for planning; the reported fee is used for the ledger.
When they disagree persistently, the fee configuration is wrong and the engine
says so rather than quietly accruing an error.
"""

from __future__ import annotations

import abc
import asyncio
import logging
from decimal import Decimal
from typing import Any, Mapping, Sequence

from triangulum.core.clock import Clock, EwmaLatency, SystemClock
from triangulum.core.config import VenueConfig
from triangulum.core.decimal_math import D, ZERO
from triangulum.core.errors import OrderRejected, VenueError
from triangulum.core.types import (
    Asset,
    Balance,
    Fill,
    Order,
    OrderStatus,
    Side,
    Symbol,
    TimeInForce,
)
from triangulum.exchanges.spec import SymbolSpec, VenueSpec, get_venue_spec
from triangulum.marketdata.book_manager import BookManager
from triangulum.marketdata.normalizer import SymbolNormalizer

logger = logging.getLogger(__name__)

__all__ = ["ExchangeAdapter", "AdapterCapabilities"]


class AdapterCapabilities:
    """What this adapter can actually do, as opposed to what the venue offers."""

    __slots__ = (
        "streaming_books", "streaming_fills", "post_only", "fok", "ioc",
        "query_by_client_id", "cancel_all", "batch_orders", "fee_in_response",
    )

    def __init__(
        self,
        *,
        streaming_books: bool = True,
        streaming_fills: bool = True,
        post_only: bool = True,
        fok: bool = True,
        ioc: bool = True,
        query_by_client_id: bool = True,
        cancel_all: bool = True,
        batch_orders: bool = False,
        fee_in_response: bool = True,
    ) -> None:
        self.streaming_books = streaming_books
        self.streaming_fills = streaming_fills
        self.post_only = post_only
        self.fok = fok
        self.ioc = ioc
        self.query_by_client_id = query_by_client_id
        self.cancel_all = cancel_all
        self.batch_orders = batch_orders
        self.fee_in_response = fee_in_response

    def supports_tif(self, tif: TimeInForce) -> bool:
        return {
            TimeInForce.GTC: True,
            TimeInForce.IOC: self.ioc,
            TimeInForce.FOK: self.fok,
            TimeInForce.GTX: self.post_only,
        }[tif]


class ExchangeAdapter(abc.ABC):
    """Base class for all venues, live and simulated."""

    def __init__(
        self,
        config: VenueConfig,
        normalizer: SymbolNormalizer,
        books: BookManager,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.name = config.name
        self.normalizer = normalizer
        self.books = books
        self.clock = clock or SystemClock()
        self.spec: VenueSpec = get_venue_spec(config.name)
        self.capabilities = AdapterCapabilities()

        self._symbol_specs: dict[str, SymbolSpec] = {}
        self._balances: dict[str, Balance] = {}
        self._open_orders: dict[str, Order] = {}
        self._connected = False

        self.submit_latency = EwmaLatency()
        self.orders_submitted = 0
        self.orders_filled = 0
        self.orders_rejected = 0
        self.errors = 0

    # -- lifecycle ---------------------------------------------------------

    @abc.abstractmethod
    async def connect(self) -> None:
        """Authenticate, load instruments, open streams."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        ...

    @abc.abstractmethod
    async def load_instruments(self) -> Sequence[Symbol]:
        """Fetch the venue's tradable instruments and register their specs."""

    # -- market data -------------------------------------------------------

    @abc.abstractmethod
    async def subscribe_books(self, symbols: Sequence[Symbol]) -> None:
        ...

    async def resnapshot(self, symbol: Symbol) -> None:
        """Re-fetch a full book snapshot. Default: no-op for snapshot-only feeds."""
        return None

    # -- trading -----------------------------------------------------------

    @abc.abstractmethod
    async def submit(self, order: Order, *, timeout_sec: float = 5.0) -> Order:
        """
        Submit and resolve an order.

        Returns the order in a state the caller can reason about. Must not
        return while the order's fate is still unknown -- on timeout, query by
        client order id and return the truth, or raise.
        """

    @abc.abstractmethod
    async def cancel(self, order: Order) -> Order:
        ...

    @abc.abstractmethod
    async def fetch_order(self, order: Order) -> Order:
        ...

    @abc.abstractmethod
    async def fetch_balances(self) -> Mapping[str, Balance]:
        ...

    async def cancel_all(self, symbol: Symbol | None = None) -> int:
        cancelled = 0
        for order in list(self._open_orders.values()):
            if symbol is not None and order.symbol.key != symbol.key:
                continue
            try:
                await self.cancel(order)
                cancelled += 1
            except VenueError:
                logger.exception("cancel_all: failed on %s", order.client_order_id)
        return cancelled

    # -- instrument rules --------------------------------------------------

    def symbol_spec(self, symbol: Symbol) -> SymbolSpec:
        spec = self._symbol_specs.get(symbol.venue_symbol)
        if spec is None:
            # Conservative fallback: assume the venue's typical minimum and a
            # coarse lot grid. Being pessimistic here means the sizing code
            # under-trades rather than getting rejected.
            spec = SymbolSpec(
                venue_symbol=symbol.venue_symbol,
                base=symbol.base.code,
                quote=symbol.quote.code,
                min_notional=self.spec.typical_min_notional_usd,
            )
            self._symbol_specs[symbol.venue_symbol] = spec
        return spec

    def register_spec(self, spec: SymbolSpec) -> None:
        self._symbol_specs[spec.venue_symbol] = spec

    def all_specs(self) -> Mapping[str, SymbolSpec]:
        return dict(self._symbol_specs)

    # -- validation --------------------------------------------------------

    def validate_order(self, order: Order) -> None:
        """
        Reject locally what the venue would reject remotely.

        A local rejection costs microseconds; a remote one costs a round trip
        plus, in the middle of a cycle, the entire opportunity. Every check here
        corresponds to a real venue error code.
        """
        spec = self.symbol_spec(order.symbol)

        if order.quantity <= 0:
            raise OrderRejected("quantity must be positive", venue=self.name, code="QTY_ZERO")
        if spec.min_quantity and order.quantity < spec.min_quantity:
            raise OrderRejected(
                f"quantity {order.quantity} below minimum {spec.min_quantity}",
                venue=self.name, code="LOT_SIZE",
            )
        if order.quantity > spec.max_quantity:
            raise OrderRejected(
                f"quantity {order.quantity} above maximum {spec.max_quantity}",
                venue=self.name, code="LOT_SIZE",
            )

        # Quantity must sit exactly on the lot grid.
        if spec.lot_step > 0:
            remainder = order.quantity % spec.lot_step
            if remainder != 0:
                raise OrderRejected(
                    f"quantity {order.quantity} is not a multiple of lot step "
                    f"{spec.lot_step} (remainder {remainder})",
                    venue=self.name, code="LOT_SIZE",
                )

        if order.price is not None and order.price > 0:
            if spec.tick_size > 0 and order.price % spec.tick_size != 0:
                raise OrderRejected(
                    f"price {order.price} is not a multiple of tick size {spec.tick_size}",
                    venue=self.name, code="PRICE_FILTER",
                )
            notional = order.quantity * order.price
            if notional < spec.min_notional:
                raise OrderRejected(
                    f"notional {notional} below minimum {spec.min_notional}",
                    venue=self.name, code="MIN_NOTIONAL",
                )

        if not self.capabilities.supports_tif(order.time_in_force):
            raise OrderRejected(
                f"{self.name} does not support {order.time_in_force.value}",
                venue=self.name, code="TIF_UNSUPPORTED",
            )

    # -- balances ----------------------------------------------------------

    def balance(self, asset: Asset) -> Balance:
        return self._balances.get(asset.code, Balance(asset=asset, free=ZERO))

    def available(self, asset: Asset) -> Decimal:
        return self.balance(asset).free

    def set_balance(self, asset: Asset, free: Decimal, locked: Decimal = ZERO) -> None:
        self._balances[asset.code] = Balance(asset=asset, free=free, locked=locked)

    def all_balances(self) -> Mapping[str, Balance]:
        return dict(self._balances)

    # -- state -------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def open_orders(self) -> Sequence[Order]:
        return list(self._open_orders.values())

    def _track(self, order: Order) -> None:
        if order.status.terminal:
            self._open_orders.pop(order.client_order_id, None)
        else:
            self._open_orders[order.client_order_id] = order

    def stats(self) -> dict[str, Any]:
        return {
            "venue": self.name,
            "connected": self._connected,
            "instruments": len(self._symbol_specs),
            "open_orders": len(self._open_orders),
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "errors": self.errors,
            "submit_latency_ms": round(self.submit_latency.mean_ms, 2),
            "submit_latency_p95_ms": round(self.submit_latency.p95_estimate_ns / 1e6, 2),
            "fill_rate": (
                self.orders_filled / self.orders_submitted
                if self.orders_submitted else 0.0
            ),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name} connected={self._connected}>"
