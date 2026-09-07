"""
Configuration.

Deliberately built on plain dataclasses plus a small validation layer rather
than pydantic. Two reasons:

1. The config object is constructed before anything else, including the logging
   setup. A dependency failure here produces the worst possible error message.
   Stdlib-only means it cannot fail for environmental reasons.

2. Trading configs need *semantic* cross-field validation ("your max cycle
   notional exceeds your equity", "you asked for maker legs on a venue with no
   maker rebate", "your capital cannot clear this venue's min-notional across
   three legs"), which is custom code either way.

Precedence, lowest to highest:
    dataclass defaults  <  config/default.yaml  <  --config file  <  environment

Secrets never live in YAML. API keys are read from the environment only, and
the loader refuses to start if it finds something that looks like a key in a
config file -- that check has saved more than one repository from a bad commit.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence, get_args, get_origin

from triangulum.core import constants as C
from triangulum.core.decimal_math import D
from triangulum.core.errors import ConfigError
from triangulum.core.types import ExecutionMode, RunMode

__all__ = [
    "Config",
    "VenueConfig",
    "StrategyConfig",
    "ExecutionConfig",
    "RiskConfig",
    "LearningConfig",
    "DashboardConfig",
    "StorageConfig",
    "OpsConfig",
    "load_config",
    "dump_config",
]


_SECRET_PATTERN = re.compile(
    r"(api[_-]?key|api[_-]?secret|secret[_-]?key|passphrase|private[_-]?key|token)",
    re.IGNORECASE,
)
# A value that looks like a real credential rather than a placeholder.
_SECRET_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9+/=_-]{24,}$")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


@dataclass(slots=True)
class VenueConfig:
    """
    One exchange connection.

    Credentials are resolved from the environment using ``{NAME}_API_KEY`` etc.
    where NAME is the uppercased venue id, so ``binance`` reads
    ``BINANCE_API_KEY`` / ``BINANCE_API_SECRET`` / ``BINANCE_API_PASSPHRASE``.
    """

    name: str
    enabled: bool = True
    sandbox: bool = False

    # Universe control. An empty ``quote_assets`` means "all discovered".
    quote_assets: tuple[str, ...] = ()
    base_assets: tuple[str, ...] = ()
    symbol_allowlist: tuple[str, ...] = ()
    symbol_denylist: tuple[str, ...] = ()
    max_symbols: int = 250

    # Book depth we maintain. Deeper costs bandwidth and CPU for very little
    # benefit: a $100 account never reaches level 5.
    book_depth: int = 20

    # Fee overrides. Leave at None to use the venue's published default tier;
    # set these once you know your actual tier, because the default is usually
    # pessimistic and the difference decides which cycles are viable.
    maker_fee_bps: float | None = None
    taker_fee_bps: float | None = None
    fee_discount_asset: str = ""      # BNB on Binance, KCS on KuCoin
    fee_discount_pct: float = 0.0

    # Connection tuning
    rest_timeout_sec: float = C.DEFAULT_HTTP_TIMEOUT_SEC
    ws_ping_interval_sec: float = C.DEFAULT_WS_PING_INTERVAL_SEC
    max_requests_per_second: float = 10.0
    max_orders_per_second: float = 5.0

    def credential_env_names(self) -> dict[str, str]:
        prefix = self.name.upper().replace("-", "_")
        return {
            "api_key": f"{prefix}_API_KEY",
            "api_secret": f"{prefix}_API_SECRET",
            "api_passphrase": f"{prefix}_API_PASSPHRASE",
        }

    def credentials(self) -> dict[str, str]:
        return {
            k: os.environ.get(v, "") for k, v in self.credential_env_names().items()
        }

    @property
    def has_credentials(self) -> bool:
        creds = self.credentials()
        return bool(creds["api_key"] and creds["api_secret"])


@dataclass(slots=True)
class StrategyConfig:
    triangular_enabled: bool = True
    polygonal_enabled: bool = True
    cross_exchange_enabled: bool = False
    statistical_enabled: bool = False

    # Cycle length bounds. 3 is the classic triangle; 4 opens up meaningfully
    # more paths at the cost of one extra leg of fees and fill risk.
    min_cycle_length: int = 3
    max_cycle_length: int = 4

    # Only start cycles from assets we are content to hold if a leg fails.
    start_assets: tuple[str, ...] = C.PREFERRED_START_ASSETS

    # Screening thresholds, before the learner's EV gate.
    min_gross_edge_bps: float = 3.0
    min_net_edge_bps: float = float(C.MIN_VIABLE_NET_EDGE_BPS)
    max_book_age_ms: float = C.MAX_BOOK_AGE_NS / 1e6

    # Re-evaluate the graph at most this often; book updates arrive far faster
    # than we could act on them and re-scanning on every tick is pure heat.
    scan_interval_ms: float = 25.0

    # Skip a cycle we just traded: repeated immediate firing on the same path
    # is a strong sign the book is stale rather than genuinely mispriced.
    cycle_cooldown_ms: float = 750.0

    # Quantization-drag budget for the instrument screen, in bps per cycle.
    #
    # This is the most consequential knob for a small account, and it is a
    # genuine trade-off with no free answer: a tight budget admits only
    # low-lot-value instruments and can leave a graph with NO cycles at all,
    # while a loose one admits BTC- and ETH-anchored cycles whose lot rounding
    # eats more than the edge. `triangulum doctor` prints both sides.
    drag_budget_bps: float = 5.0
    enforce_lot_value_screen: bool = True

    max_concurrent_cycles: int = 2


@dataclass(slots=True)
class ExecutionConfig:
    mode: str = ExecutionMode.ADAPTIVE.value

    cycle_budget_ms: float = C.DEFAULT_CYCLE_BUDGET_MS
    leg_timeout_ms: float = 250.0

    # Fraction of top-of-book size we will consume on any single leg.
    max_book_consumption: float = float(C.MAX_TOP_OF_BOOK_CONSUMPTION)
    # How many levels the sizing walk may descend before giving up.
    max_levels_to_walk: int = 8

    # Aggressiveness of marketable limit prices, in ticks past the touch.
    # Zero means "at the touch", which under-fills; 1-2 ticks buys fill
    # certainty for a fraction of a bp and is almost always correct.
    taker_price_offset_ticks: int = 2
    maker_price_offset_ticks: int = 1

    # Post-only leg handling
    maker_leg_timeout_ms: float = 1500.0
    maker_max_requeues: int = 2

    # Unwinding
    unwind_enabled: bool = True
    unwind_aggressiveness_ticks: int = 5
    unwind_max_attempts: int = 3

    # Sizing
    capital_fraction_per_cycle: float = 0.95   # of the available start asset
    kelly_fraction_cap: float = 0.25
    min_notional_buffer: float = 1.15          # 15% above venue minimum

    dry_run_orders: bool = False               # log orders, never submit


@dataclass(slots=True)
class RiskConfig:
    max_daily_loss_pct: float = float(C.DEFAULT_MAX_DAILY_LOSS_PCT)
    max_drawdown_pct: float = float(C.DEFAULT_MAX_DRAWDOWN_PCT)
    max_consecutive_losses: int = C.DEFAULT_MAX_CONSECUTIVE_LOSSES
    max_cycle_notional_pct: float = float(C.DEFAULT_MAX_CYCLE_NOTIONAL_PCT)
    max_venue_exposure_pct: float = 100.0
    max_asset_exposure_pct: float = 60.0

    max_orders_per_minute: int = 120
    max_cycles_per_minute: int = 30

    # Data-integrity breakers
    max_book_staleness_ms: float = 500.0
    reconcile_tolerance_bps: float = float(C.DEFAULT_RECONCILE_TOLERANCE_BPS)
    reconcile_interval_sec: float = 30.0

    # Error-rate breaker: trip if this many venue errors occur in the window.
    error_burst_threshold: int = 12
    error_burst_window_sec: float = 60.0

    # Cooldown after a breaker opens before a half-open probe is allowed.
    circuit_cooldown_sec: float = 120.0

    halt_on_unwind_failure: bool = True
    halt_on_reconcile_mismatch: bool = True


@dataclass(slots=True)
class LearningConfig:
    enabled: bool = True

    # Fill-probability model (FTRL-Proximal logistic regression)
    ftrl_alpha: float = 0.1
    ftrl_beta: float = 1.0
    ftrl_l1: float = 0.5
    ftrl_l2: float = 1.0
    feature_hash_bits: int = 18       # 262144 buckets; plenty, collisions rare

    # Slippage model (online ridge)
    slippage_lr: float = 0.02
    slippage_l2: float = 0.01

    # EV gate
    min_expected_value_bps: float = 0.5
    ev_confidence_multiplier: float = 1.0    # require EV > k * sigma
    exploration_floor: float = C.EXPLORATION_FLOOR
    min_samples_before_trust: int = C.MIN_SAMPLES_BEFORE_TRUST

    # Bandit over execution configurations
    bandit_enabled: bool = True
    bandit_prior_alpha: float = 1.0
    bandit_prior_beta: float = 1.0
    bandit_decay: float = 0.999        # forget slowly; regimes change

    # Drift detection
    drift_enabled: bool = True
    drift_delta: float = 0.002         # ADWIN confidence
    drift_min_samples: int = 100

    # Persistence
    checkpoint_interval_sec: float = 300.0
    model_dir: str = C.DEFAULT_MODEL_DIR


@dataclass(slots=True)
class DashboardConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8787
    broadcast_interval_ms: float = 250.0
    max_log_lines: int = 500
    max_opportunity_rows: int = 200
    auth_token: str = ""               # from TRIANGULUM_DASHBOARD_TOKEN
    allow_kill_switch: bool = True


@dataclass(slots=True)
class StorageConfig:
    db_path: str = C.DEFAULT_DB_PATH
    recording_dir: str = C.DEFAULT_RECORDING_DIR
    record_books: bool = False         # high volume; enable when collecting data
    record_opportunities: bool = True
    record_cycles: bool = True
    retention_days: int = 90
    flush_interval_sec: float = 5.0


@dataclass(slots=True)
class OpsConfig:
    log_level: str = "INFO"
    log_json: bool = False
    log_file: str = ""
    heartbeat_interval_sec: float = 10.0
    metrics_enabled: bool = True
    alert_webhook_url: str = ""
    alert_on_severity: str = "CRITICAL"
    telegram_bot_token: str = ""       # from TELEGRAM_BOT_TOKEN
    telegram_chat_id: str = ""


# --------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Config:
    mode: str = RunMode.PAPER.value

    # Capital. ``base_currency`` is the unit the equity curve is denominated in.
    base_currency: str = "USDT"
    initial_capital: float = 100.0

    # The target you asked for, tracked honestly. The engine does not chase it
    # -- chasing a return target by relaxing the EV gate is precisely how a
    # positive-expectancy system is turned into a negative one -- but every
    # panel measures against it so the gap is never in doubt.
    daily_target_bps: float = 100.0        # 1.00% / day
    monthly_target_pct: float = 30.0

    venues: tuple[VenueConfig, ...] = ()
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    ops: OpsConfig = field(default_factory=OpsConfig)

    config_path: str = ""

    # -- derived -----------------------------------------------------------

    @property
    def run_mode(self) -> RunMode:
        return RunMode(self.mode)

    @property
    def enabled_venues(self) -> tuple[VenueConfig, ...]:
        return tuple(v for v in self.venues if v.enabled)

    @property
    def venue_names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.enabled_venues)

    def venue(self, name: str) -> VenueConfig:
        for v in self.venues:
            if v.name == name:
                return v
        raise ConfigError(f"venue {name!r} not configured", known=list(self.venue_names))

    @property
    def capital(self) -> Decimal:
        return D(self.initial_capital)

    # -- validation --------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Returns a list of human-readable warnings. Raises ConfigError on
        anything that makes the configuration unrunnable.
        """
        warnings: list[str] = []

        try:
            RunMode(self.mode)
        except ValueError:
            raise ConfigError(
                f"invalid mode {self.mode!r}",
                valid=[m.value for m in RunMode],
            ) from None

        try:
            ExecutionMode(self.execution.mode)
        except ValueError:
            raise ConfigError(
                f"invalid execution mode {self.execution.mode!r}",
                valid=[m.value for m in ExecutionMode],
            ) from None

        if not self.enabled_venues:
            raise ConfigError("no venues enabled")

        if self.initial_capital <= 0:
            raise ConfigError("initial_capital must be positive")

        s = self.strategy
        if s.min_cycle_length < 3:
            raise ConfigError("min_cycle_length must be >= 3 (a 2-cycle is a round trip)")
        if s.max_cycle_length < s.min_cycle_length:
            raise ConfigError("max_cycle_length < min_cycle_length")
        if s.max_cycle_length > 6:
            warnings.append(
                "max_cycle_length > 6: each additional leg multiplies fee drag and "
                "fill risk; cycles this long almost never survive the EV gate."
            )
        if s.min_net_edge_bps < float(C.MIN_VIABLE_NET_EDGE_BPS):
            warnings.append(
                f"min_net_edge_bps={s.min_net_edge_bps} is below the empirical "
                f"viability floor of {C.MIN_VIABLE_NET_EDGE_BPS}; expect adverse "
                "selection to dominate."
            )

        e = self.execution
        if not 0 < e.capital_fraction_per_cycle <= 1.0:
            raise ConfigError("capital_fraction_per_cycle must be in (0, 1]")
        if e.max_book_consumption > 0.5:
            warnings.append(
                "max_book_consumption > 0.5: consuming more than half the touch "
                "moves the price against you and telegraphs the strategy."
            )
        if e.cycle_budget_ms > 3000:
            warnings.append(
                "cycle_budget_ms > 3000: opportunities decay with a half-life "
                "measured in tens of milliseconds; a 3s budget mostly buys you "
                "the right to fill at a worse price."
            )

        r = self.risk
        if r.max_cycle_notional_pct > 100:
            raise ConfigError("max_cycle_notional_pct > 100")
        if r.max_daily_loss_pct >= 100:
            raise ConfigError("max_daily_loss_pct >= 100 is not a limit")

        # Cross-field economics
        if self.run_mode is RunMode.LIVE and self.initial_capital < 500:
            warnings.append(
                f"live mode with {self.initial_capital:.2f} {self.base_currency}: "
                "min-notional quantization will consume a large fraction of the "
                "theoretical edge. See docs/EXPECTATIONS.md for the arithmetic."
            )

        if self.daily_target_bps > 20:
            warnings.append(
                f"daily_target_bps={self.daily_target_bps:.0f} "
                f"({self.daily_target_bps / 100:.2f}%/day compounds to "
                f"{(1 + self.daily_target_bps / 10000) ** 365 - 1:,.0%}/yr). "
                "Tracked as a target, never as a constraint on the EV gate."
            )

        if self.execution.mode in (ExecutionMode.MTT.value, ExecutionMode.TMT.value):
            for v in self.enabled_venues:
                if v.maker_fee_bps is not None and v.maker_fee_bps >= (v.taker_fee_bps or 10):
                    warnings.append(
                        f"{v.name}: maker fee is not better than taker; maker-leg "
                        "execution adds fill risk with no fee benefit here."
                    )

        return warnings

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def redacted(self) -> dict[str, Any]:
        """Config safe to log or ship to the dashboard."""
        data = self.to_dict()
        data.pop("config_path", None)
        for section in ("dashboard", "ops"):
            for key in list(data.get(section, {})):
                if _SECRET_PATTERN.search(key):
                    data[section][key] = "***" if data[section][key] else ""
        return data


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    """
    Minimal YAML reader.

    Tries PyYAML, then falls back to JSON, then to a tiny nested-mapping parser
    that handles the subset this project's configs actually use. The fallback
    exists so ``triangulum doctor`` can still tell you what is wrong with your
    config on a box where the dependency install failed.
    """
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return json.loads(text)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Indentation-based parser for scalars, nested maps, and lists of maps."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip() if not raw_line.strip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        content = line.strip()

        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            stack = [(-1, root)]
        parent = stack[-1][1]

        if content.startswith("- "):
            item_text = content[2:].strip()
            if not isinstance(parent, list):
                continue
            if ":" in item_text:
                obj: dict[str, Any] = {}
                key, _, val = item_text.partition(":")
                obj[key.strip()] = _coerce_scalar(val.strip())
                parent.append(obj)
                stack.append((indent, obj))
            else:
                parent.append(_coerce_scalar(item_text))
            continue

        if ":" not in content:
            continue
        key, _, val = content.partition(":")
        key = key.strip()
        val = val.strip()
        if not val:
            child: dict[str, Any] | list[Any] = {}
            if isinstance(parent, dict):
                parent[key] = child
            stack.append((indent, child))
            # A following "- " line converts it to a list.
            stack.append((indent, child))
            stack.pop()
        elif val.startswith("["):
            inner = val.strip("[]").strip()
            items = [_coerce_scalar(x.strip()) for x in inner.split(",") if x.strip()]
            if isinstance(parent, dict):
                parent[key] = items
        else:
            if isinstance(parent, dict):
                parent[key] = _coerce_scalar(val)
    return _fix_empty_lists(root)


def _fix_empty_lists(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _fix_empty_lists(v) for k, v in node.items()}
    return node


def _coerce_scalar(value: str) -> Any:
    v = value.strip().strip('"').strip("'")
    low = v.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    try:
        if "." in v or "e" in low:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _scan_for_secrets(data: Mapping[str, Any], path: str = "") -> list[str]:
    """Refuse to load a config that appears to contain live credentials."""
    found: list[str] = []
    for key, value in data.items():
        here = f"{path}.{key}" if path else str(key)
        if isinstance(value, Mapping):
            found.extend(_scan_for_secrets(value, here))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, Mapping):
                    found.extend(_scan_for_secrets(item, f"{here}[{i}]"))
        elif isinstance(value, str) and _SECRET_PATTERN.search(str(key)):
            if _SECRET_VALUE_PATTERN.match(value):
                found.append(here)
    return found


def _build(cls: type, data: Mapping[str, Any]) -> Any:
    """Recursively construct a dataclass from a mapping, coercing types."""
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            continue
        f = known[key]
        kwargs[key] = _coerce_field(f.type, value)
    return cls(**kwargs)


def _coerce_field(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        if args and is_dataclass(args[0]) and isinstance(value, list):
            return tuple(_build(args[0], item) for item in value)
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return value
    if isinstance(annotation, str):
        # Deferred annotations: match on the textual type.
        if annotation.startswith("tuple[") and isinstance(value, list):
            if "VenueConfig" in annotation:
                return tuple(_build(VenueConfig, item) for item in value)
            return tuple(value)
        for section_cls in (
            StrategyConfig, ExecutionConfig, RiskConfig, LearningConfig,
            DashboardConfig, StorageConfig, OpsConfig, VenueConfig,
        ):
            if annotation == section_cls.__name__ and isinstance(value, Mapping):
                return _build(section_cls, value)
        return value
    if is_dataclass(annotation) and isinstance(value, Mapping):
        return _build(annotation, value)
    return value


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


_ENV_PREFIX = "TRIANGULUM_"


def _env_overrides() -> dict[str, Any]:
    """
    ``TRIANGULUM_RISK__MAX_DAILY_LOSS_PCT=2.0`` -> ``risk.max_daily_loss_pct``.

    Double underscore is the section separator. Used heavily in containerized
    deployments where mounting a config file is more friction than it is worth.
    """
    out: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        path = key[len(_ENV_PREFIX):].lower()
        if path in ("arm_live", "dashboard_token"):
            continue
        parts = path.split("__")
        cursor = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce_scalar(raw)
    return out


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    apply_env: bool = True,
) -> tuple[Config, list[str]]:
    """
    Build a :class:`Config`, returning it alongside any validation warnings.

    The caller is expected to log the warnings. They are returned rather than
    logged here because logging is not configured yet at this point in startup.
    """
    data: dict[str, Any] = {}

    default_path = Path("config/default.yaml")
    if default_path.exists():
        data = _deep_merge(data, _read_yaml(default_path))

    resolved = Path(path) if path else None
    if resolved is not None:
        if not resolved.exists():
            raise ConfigError(f"config file not found: {resolved}")
        file_data = _read_yaml(resolved)
        secrets = _scan_for_secrets(file_data)
        if secrets:
            raise ConfigError(
                "config file appears to contain live credentials; move them to "
                "environment variables and never commit them",
                keys=secrets,
                file=str(resolved),
            )
        data = _deep_merge(data, file_data)

    if apply_env:
        data = _deep_merge(data, _env_overrides())
    if overrides:
        data = _deep_merge(data, overrides)

    cfg: Config = _build(Config, data)
    if resolved is not None:
        cfg.config_path = str(resolved)

    # Secrets that legitimately come from the environment.
    if not cfg.dashboard.auth_token:
        cfg.dashboard.auth_token = os.environ.get("TRIANGULUM_DASHBOARD_TOKEN", "")
    if not cfg.ops.telegram_bot_token:
        cfg.ops.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not cfg.ops.telegram_chat_id:
        cfg.ops.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    warnings = cfg.validate()
    return cfg, warnings


def dump_config(cfg: Config, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(cfg.to_json(), encoding="utf-8")
