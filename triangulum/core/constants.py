"""Project-wide constants. No logic, no imports beyond stdlib."""

from __future__ import annotations

from decimal import Decimal

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
APP_NAME = "triangulum"
USER_AGENT = "Triangulum/1.0 (+https://github.com/kiwiplusapp/triangulum)"

# --------------------------------------------------------------------------
# Basis points
# --------------------------------------------------------------------------
BPS = Decimal("0.0001")
ONE_PERCENT_BPS = Decimal(100)

# --------------------------------------------------------------------------
# Economic reality checks
#
# These are not tuning knobs -- they encode facts about the venues. They exist
# so the engine can refuse to run a configuration that cannot possibly work,
# rather than discovering it $40 later.
# --------------------------------------------------------------------------

# Below this net edge, the expected value after adverse selection is negative
# regardless of what the naive arithmetic says. Derived empirically: cycles
# under ~1.5 bps are dominated by queue position and latency, both of which a
# retail participant loses.
MIN_VIABLE_NET_EDGE_BPS = Decimal("1.5")

# A cycle whose oldest book is staler than this is not an opportunity, it is a
# memory of one.
MAX_BOOK_AGE_NS = 250_000_000  # 250ms

# Hard ceiling on how long a cycle may take end-to-end before we stop opening
# legs and start unwinding.
DEFAULT_CYCLE_BUDGET_MS = 900

# Fraction of top-of-book size we are willing to consume on one leg. Taking
# more than this moves the price against us and signals our presence.
MAX_TOP_OF_BOOK_CONSUMPTION = Decimal("0.35")

# --------------------------------------------------------------------------
# Risk defaults (overridable in config, but these are the safe floor)
# --------------------------------------------------------------------------
DEFAULT_MAX_DAILY_LOSS_PCT = Decimal("3.0")
DEFAULT_MAX_DRAWDOWN_PCT = Decimal("10.0")
DEFAULT_MAX_CONSECUTIVE_LOSSES = 8
DEFAULT_MAX_CYCLE_NOTIONAL_PCT = Decimal("35.0")   # of equity
DEFAULT_RECONCILE_TOLERANCE_BPS = Decimal("5")

# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------
# Cycles observed before the fill model's output is trusted over the prior.
MIN_SAMPLES_BEFORE_TRUST = 200
# Exploration floor: even a well-fit model keeps taking some marginal trades,
# otherwise the training distribution collapses onto its own predictions and
# the model can never learn that it was wrong.
EXPLORATION_FLOOR = 0.05

# --------------------------------------------------------------------------
# Networking
# --------------------------------------------------------------------------
DEFAULT_HTTP_TIMEOUT_SEC = 8.0
DEFAULT_WS_PING_INTERVAL_SEC = 20.0
DEFAULT_WS_PING_TIMEOUT_SEC = 10.0
MAX_RECONNECT_BACKOFF_SEC = 60.0
RECONNECT_BACKOFF_BASE_SEC = 1.0

# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
DEFAULT_DB_PATH = "data/triangulum.db"
DEFAULT_MODEL_DIR = "data/models"
DEFAULT_RECORDING_DIR = "data/recordings"

# --------------------------------------------------------------------------
# Live-trading triple lock
#
# All three must be satisfied simultaneously. They are independent on purpose:
# a typo in a config file, an inherited shell environment, and a stale file on
# disk are three different accidents, and no single one of them should be able
# to move real money.
# --------------------------------------------------------------------------
LIVE_ENV_VAR = "TRIANGULUM_ARM_LIVE"
LIVE_ENV_VALUE = "I_ACCEPT_FULL_LOSS_OF_CAPITAL"
LIVE_ACK_FILENAME = ".triangulum-live-ack"
LIVE_ACK_REQUIRED_TEXT = "I have read docs/EXPECTATIONS.md and accept total loss."

# --------------------------------------------------------------------------
# Well-known assets
# --------------------------------------------------------------------------
STABLECOINS = frozenset({
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDP", "PYUSD", "USDE", "EURC",
})
FIAT = frozenset({
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SGD",
    "HKD", "TRY", "BRL", "MXN", "ARS", "ZAR", "PLN", "SEK", "NOK", "DKK",
})
METALS = frozenset({"XAU", "XAG", "XPT", "XPD"})

# Assets that make good cycle anchors: deep, cheap to hold, low volatility.
PREFERRED_START_ASSETS = ("USDT", "USDC", "USD", "EUR", "BTC", "ETH")
