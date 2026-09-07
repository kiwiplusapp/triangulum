"""
Symbol normalization across venues.

The same economic pair has a different name on every venue, and the differences
are not merely cosmetic:

    Binance   BTCUSDT       no separator, base first
    Kraken    XXBTZUSD      legacy X/Z asset-class prefixes, BTC spelled XBT
    Coinbase  BTC-USD       hyphen
    KuCoin    BTC-USDT      hyphen, different quote
    OKX       BTC-USDT      hyphen
    Bybit     BTCUSDT       no separator
    OANDA     BTC_USD       underscore

Without a separator you cannot split ``BTCUSDT`` into base and quote by string
manipulation alone -- is it ``BTC``/``USDT`` or ``BTCUSD``/``T``? The only
correct approach is to match against a known set of quote assets, longest
first, and even then you need the venue's own instrument listing to
disambiguate. This module does both: a longest-match splitter for bootstrapping
and an authoritative registry populated from each venue's instrument endpoint.

The Kraken asset-code rules deserve special mention because they cause real
bugs: Kraken prefixes crypto with ``X`` and fiat with ``Z`` on *legacy* pairs
only, spells Bitcoin ``XBT``, and Doge ``XDG``. A cycle built from
``XXBTZUSD`` + ``XETHXXBT`` + ``XETHZUSD`` is a perfectly good triangle that a
naive parser will not even recognise as connected.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from triangulum.core.constants import FIAT, METALS, STABLECOINS
from triangulum.core.types import Asset, AssetClass, Symbol

__all__ = [
    "SymbolNormalizer",
    "classify_asset",
    "canonical_asset_code",
    "KRAKEN_ASSET_ALIASES",
    "QUOTE_ASSET_PRIORITY",
]


# Quote assets in longest-first order. Order matters enormously: if ``USD``
# comes before ``USDT`` then ``BTCUSDT`` splits as ``BTCUSD``/``T``.
QUOTE_ASSET_PRIORITY: tuple[str, ...] = (
    "FDUSD", "TUSD", "BUSD", "USDC", "USDT", "USDE", "PYUSD", "EURC", "DAI",
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "TRY", "BRL", "ARS",
    "BTC", "XBT", "ETH", "BNB", "SOL", "DOT", "TRX",
)

# Kraken's legacy asset codes -> canonical.
KRAKEN_ASSET_ALIASES: Mapping[str, str] = {
    "XBT": "BTC", "XXBT": "BTC",
    "XDG": "DOGE", "XXDG": "DOGE",
    "XETH": "ETH", "XETC": "ETC", "XLTC": "LTC", "XMLN": "MLN",
    "XREP": "REP", "XXLM": "XLM", "XXMR": "XMR", "XXRP": "XRP",
    "XZEC": "ZEC", "XICN": "ICN", "XNMC": "NMC", "XXVN": "XVN",
    "ZUSD": "USD", "ZEUR": "EUR", "ZGBP": "GBP", "ZJPY": "JPY",
    "ZCAD": "CAD", "ZAUD": "AUD", "ZCHF": "CHF",
}

# Venues that report the same asset under different tickers.
GENERIC_ASSET_ALIASES: Mapping[str, str] = {
    "XBT": "BTC",
    "BCHABC": "BCH",
    "BCC": "BCH",       # pre-2018 Binance name for Bitcoin Cash
    "MIOTA": "IOTA",
    "WBTC": "WBTC",     # explicitly NOT BTC: wrapped, different settlement risk
}


def canonical_asset_code(raw: str, venue: str = "") -> str:
    """Map a venue asset code to the canonical one used across the engine."""
    code = raw.strip().upper()
    if venue == "kraken" and code in KRAKEN_ASSET_ALIASES:
        return KRAKEN_ASSET_ALIASES[code]
    # Kraken sometimes returns the legacy code even on newer endpoints.
    if code in KRAKEN_ASSET_ALIASES and len(code) == 4 and code[0] in ("X", "Z"):
        return KRAKEN_ASSET_ALIASES[code]
    return GENERIC_ASSET_ALIASES.get(code, code)


def classify_asset(code: str) -> AssetClass:
    c = code.upper()
    if c in STABLECOINS:
        return AssetClass.STABLECOIN
    if c in FIAT:
        return AssetClass.FIAT
    if c in METALS:
        return AssetClass.METAL
    return AssetClass.CRYPTO


class SymbolNormalizer:
    """
    Bidirectional map between venue symbol strings and canonical
    :class:`Symbol` objects.

    Populated authoritatively from each venue's instrument listing at startup.
    :meth:`guess` exists only for bootstrapping and for parsing recorded data
    whose instrument metadata was not captured.
    """

    def __init__(self) -> None:
        self._by_venue_symbol: dict[tuple[str, str], Symbol] = {}
        self._by_canonical: dict[tuple[str, str], Symbol] = {}
        self._assets: dict[str, Asset] = {}

    # -- registration ------------------------------------------------------

    def asset(self, code: str, venue: str = "") -> Asset:
        """Interned Asset. Interning matters: assets are dict keys everywhere."""
        canonical = canonical_asset_code(code, venue)
        existing = self._assets.get(canonical)
        if existing is not None:
            return existing
        created = Asset(canonical, classify_asset(canonical))
        self._assets[canonical] = created
        return created

    def register(
        self,
        venue: str,
        venue_symbol: str,
        base_code: str,
        quote_code: str,
    ) -> Symbol:
        base = self.asset(base_code, venue)
        quote = self.asset(quote_code, venue)
        symbol = Symbol(base=base, quote=quote, venue=venue, venue_symbol=venue_symbol)
        self._by_venue_symbol[(venue, venue_symbol)] = symbol
        self._by_venue_symbol[(venue, venue_symbol.upper())] = symbol
        self._by_canonical[(venue, symbol.canonical)] = symbol
        return symbol

    def register_many(
        self, venue: str, instruments: Iterable[tuple[str, str, str]]
    ) -> list[Symbol]:
        return [self.register(venue, vs, b, q) for vs, b, q in instruments]

    # -- lookup ------------------------------------------------------------

    def lookup(self, venue: str, venue_symbol: str) -> Symbol | None:
        sym = self._by_venue_symbol.get((venue, venue_symbol))
        if sym is not None:
            return sym
        return self._by_venue_symbol.get((venue, venue_symbol.upper()))

    def canonical(self, venue: str, pair: str) -> Symbol | None:
        return self._by_canonical.get((venue, pair.upper()))

    def resolve(self, venue: str, venue_symbol: str) -> Symbol:
        """Lookup, falling back to :meth:`guess` and registering the result."""
        found = self.lookup(venue, venue_symbol)
        if found is not None:
            return found
        base, quote = self.guess(venue_symbol, venue)
        return self.register(venue, venue_symbol, base, quote)

    def symbols_for_venue(self, venue: str) -> list[Symbol]:
        return [s for (v, _), s in self._by_canonical.items() if v == venue]

    def all_assets(self) -> list[Asset]:
        return list(self._assets.values())

    def find_pair(self, venue: str, a: Asset, b: Asset) -> Symbol | None:
        """
        The symbol connecting two assets on a venue, in either orientation.

        The graph layer calls this for every candidate edge, so it is a plain
        dict hit rather than a scan.
        """
        direct = self._by_canonical.get((venue, f"{a.code}/{b.code}"))
        if direct is not None:
            return direct
        return self._by_canonical.get((venue, f"{b.code}/{a.code}"))

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def guess(venue_symbol: str, venue: str = "") -> tuple[str, str]:
        """
        Split a venue symbol into (base, quote) without an instrument listing.

        Separator-delimited formats are unambiguous. For concatenated formats we
        try quote assets longest-first. Kraken's legacy 8-character codes get
        their own path because they are fixed-width, not separator-delimited.
        """
        s = venue_symbol.strip().upper()

        for sep in ("-", "/", "_", ":"):
            if sep in s:
                left, _, right = s.partition(sep)
                return canonical_asset_code(left, venue), canonical_asset_code(right, venue)

        # Kraken legacy: XXBTZUSD, XETHXXBT -> fixed 4+4.
        if venue == "kraken" and len(s) == 8 and s[0] in "XZ" and s[4] in "XZ":
            return (
                canonical_asset_code(s[:4], venue),
                canonical_asset_code(s[4:], venue),
            )

        for quote in QUOTE_ASSET_PRIORITY:
            if s.endswith(quote) and len(s) > len(quote):
                base = s[: -len(quote)]
                if base:
                    return canonical_asset_code(base, venue), canonical_asset_code(quote, venue)

        # Last resort: assume a 3-character quote. Wrong often enough that the
        # caller should treat an unregistered symbol as untradeable.
        if len(s) > 3:
            return canonical_asset_code(s[:-3], venue), canonical_asset_code(s[-3:], venue)
        return s, ""

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_canonical)

    def stats(self) -> dict[str, object]:
        venues: dict[str, int] = {}
        for (venue, _) in self._by_canonical:
            venues[venue] = venues.get(venue, 0) + 1
        return {
            "total_symbols": len(self._by_canonical),
            "total_assets": len(self._assets),
            "by_venue": venues,
        }
