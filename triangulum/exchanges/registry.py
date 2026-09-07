"""
Adapter registry and factory.

Keeps the wiring in one place so ``engine.py`` never imports a venue module
directly. Adding a venue is: write the adapter, add one line here.

The registry also implements the paper-mode substitution that makes the whole
system testable: in PAPER mode the *real* adapter is still constructed and used
for market data and instrument metadata (so the lot steps, tick sizes and
min-notionals are the venue's true ones), while order handling is routed to a
:class:`SimulatedExchange` bound to the same books. You get real market
microstructure with synthetic fills -- which is the only paper mode worth
running, because a simulation on invented instrument rules tells you nothing
about whether your orders would have been accepted.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Callable, Mapping, Sequence

from triangulum.core.clock import Clock
from triangulum.core.config import Config, VenueConfig
from triangulum.core.errors import ConfigError, UnsupportedVenueError
from triangulum.core.types import RunMode
from triangulum.exchanges.base import ExchangeAdapter
from triangulum.exchanges.simulated import SimulatedExchange, SimulationParams
from triangulum.marketdata.book_manager import BookManager
from triangulum.marketdata.normalizer import SymbolNormalizer

logger = logging.getLogger(__name__)

__all__ = ["ADAPTERS", "build_adapters", "build_adapter", "available_venues"]


def _binance():
    from triangulum.exchanges.binance import BinanceAdapter
    return BinanceAdapter


def _kraken():
    from triangulum.exchanges.kraken import KrakenAdapter
    return KrakenAdapter


def _okx():
    from triangulum.exchanges.okx import OkxAdapter
    return OkxAdapter


def _oanda():
    from triangulum.exchanges.oanda_fx import OandaAdapter
    return OandaAdapter


def _alpaca():
    from triangulum.exchanges.alpaca import AlpacaAdapter
    return AlpacaAdapter


def _generic(venue: str) -> Callable[[], type]:
    def loader():
        from triangulum.exchanges.generic import make_generic_adapter
        return make_generic_adapter(venue)
    return loader


# Lazy loaders: importing every venue module at startup pulls in aiohttp even
# for a pure-backtest run, which should need nothing but the standard library.
ADAPTERS: Mapping[str, Callable[[], type]] = {
    "binance": _binance,
    "kraken": _kraken,
    "okx": _okx,
    "oanda": _oanda,
    "alpaca": _alpaca,
    "bybit": _generic("bybit"),
    "kucoin": _generic("kucoin"),
    "coinbase": _generic("coinbase"),
    "gateio": _generic("gateio"),
    "mexc": _generic("mexc"),
}


def available_venues() -> list[str]:
    return sorted(ADAPTERS)


def build_adapter(
    venue_config: VenueConfig,
    normalizer: SymbolNormalizer,
    books: BookManager,
    *,
    clock: Clock | None = None,
    mode: RunMode = RunMode.PAPER,
    initial_balances: Mapping[str, Decimal] | None = None,
    simulation: SimulationParams | None = None,
) -> ExchangeAdapter:
    """
    Construct the adapter for one venue, honouring the run mode.

    BACKTEST -- pure simulation; no network is touched at all.
    PAPER    -- real market data from the venue, simulated order handling.
    LIVE     -- the real adapter end to end.
    """
    name = venue_config.name

    if mode is RunMode.BACKTEST:
        return SimulatedExchange(
            venue_config, normalizer, books,
            clock=clock, params=simulation,
            initial_balances=initial_balances,
            source_venue=name,
        )

    loader = ADAPTERS.get(name)
    if loader is None:
        raise UnsupportedVenueError(
            f"no adapter for venue {name!r}", known=available_venues()
        )

    adapter_cls = loader()

    if mode is RunMode.LIVE:
        if not venue_config.has_credentials:
            raise ConfigError(
                f"live mode requires credentials for {name}; set "
                + " and ".join(venue_config.credential_env_names().values())
            )
        return adapter_cls(venue_config, normalizer, books, clock=clock)

    # PAPER: real adapter drives market data, simulator handles orders.
    live = adapter_cls(venue_config, normalizer, books, clock=clock)
    sim = SimulatedExchange(
        venue_config, normalizer, books,
        clock=clock, params=simulation,
        initial_balances=initial_balances,
        source_venue=name,
    )
    return _PaperAdapter(live, sim)


class _PaperAdapter(ExchangeAdapter):
    """
    Composite adapter: market data from the venue, fills from the simulator.

    Deliberately not a subclass of either. It is a delegating facade, so a
    method that is neither clearly market-data nor clearly trading fails loudly
    rather than silently picking the wrong half.
    """

    def __init__(self, live: ExchangeAdapter, sim: SimulatedExchange) -> None:
        super().__init__(live.config, live.normalizer, live.books, clock=live.clock)
        self.live = live
        self.sim = sim
        self.capabilities = sim.capabilities
        self.name = live.name

    async def connect(self) -> None:
        await self.live.connect()
        await self.sim.connect()
        self._connected = True

    async def disconnect(self) -> None:
        await self.live.disconnect()
        await self.sim.disconnect()
        self._connected = False

    async def load_instruments(self):
        symbols = await self.live.load_instruments()
        # The simulator must use the venue's real trading rules, or paper mode
        # tests a fantasy instrument universe.
        for spec in self.live.all_specs().values():
            self.sim.register_spec(spec)
            self.register_spec(spec)
        return symbols

    async def subscribe_books(self, symbols):
        return await self.live.subscribe_books(symbols)

    async def resnapshot(self, symbol):
        return await self.live.resnapshot(symbol)

    async def submit(self, order, *, timeout_sec: float = 5.0):
        return await self.sim.submit(order, timeout_sec=timeout_sec)

    async def cancel(self, order):
        return await self.sim.cancel(order)

    async def fetch_order(self, order):
        return await self.sim.fetch_order(order)

    async def fetch_balances(self):
        return await self.sim.fetch_balances()

    def set_balance(self, asset, free, locked=Decimal(0)) -> None:
        self.sim.set_balance(asset, free, locked)

    def balance(self, asset):
        return self.sim.balance(asset)

    def all_balances(self):
        return self.sim.all_balances()

    def symbol_spec(self, symbol):
        return self.live.symbol_spec(symbol)

    def step_resting(self, elapsed_sec: float):
        return self.sim.step_resting(elapsed_sec)

    def stats(self) -> dict:
        return {
            "venue": self.name,
            "mode": "paper",
            "market_data": self.live.stats(),
            "simulator": self.sim.stats(),
        }


def build_adapters(
    config: Config,
    normalizer: SymbolNormalizer,
    books: BookManager,
    *,
    clock: Clock | None = None,
    simulation: SimulationParams | None = None,
) -> dict[str, ExchangeAdapter]:
    """Build every enabled venue's adapter."""
    adapters: dict[str, ExchangeAdapter] = {}
    mode = config.run_mode

    # Seed the whole starting balance on the first venue. Splitting capital
    # across venues at $100 would leave each leg below min-notional, which the
    # capital-adequacy check would then reject -- correctly.
    seeded = False
    for venue_config in config.enabled_venues:
        balances = None
        if mode is not RunMode.LIVE and not seeded:
            balances = {config.base_currency: config.capital}
            seeded = True
        adapters[venue_config.name] = build_adapter(
            venue_config, normalizer, books,
            clock=clock, mode=mode,
            initial_balances=balances,
            simulation=simulation,
        )
        logger.info("built %s adapter for %s", mode.value, venue_config.name)
    return adapters
