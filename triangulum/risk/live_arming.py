"""
The live-trading triple lock.

Three independent conditions must all hold before this engine will send an order
that moves real money. They are independent by design: a typo in a config file,
an inherited shell environment, and a stale file on disk are three different
accidents, and no single one of them should be sufficient.

    1. ``mode: live`` in the configuration
    2. ``TRIANGULUM_ARM_LIVE=I_ACCEPT_FULL_LOSS_OF_CAPITAL`` in the environment
    3. A ``.triangulum-live-ack`` file containing the exact acknowledgment text

The third exists because the first two can both be set by a script. Writing the
file is a deliberate act by a person who has read what the file says.

There is no ``--force``, no ``--yes``, and no programmatic bypass. If you want
to remove this, you have to edit this file, and the diff will say so.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from triangulum.core import constants as C
from triangulum.core.errors import LiveTradingNotArmed
from triangulum.core.types import RunMode

logger = logging.getLogger(__name__)

__all__ = ["LiveArmingCheck", "check_live_arming", "write_acknowledgment"]


@dataclass(slots=True)
class LiveArmingCheck:
    config_mode_is_live: bool = False
    environment_variable_set: bool = False
    acknowledgment_file_valid: bool = False
    failures: list[str] = field(default_factory=list)

    @property
    def armed(self) -> bool:
        return (
            self.config_mode_is_live
            and self.environment_variable_set
            and self.acknowledgment_file_valid
        )

    def raise_if_not_armed(self) -> None:
        if not self.armed:
            raise LiveTradingNotArmed(
                "live trading is not armed; " + "; ".join(self.failures),
                locks_passed=sum([
                    self.config_mode_is_live,
                    self.environment_variable_set,
                    self.acknowledgment_file_valid,
                ]),
            )

    def report(self) -> str:
        lines = ["Live-trading arming status:"]
        for label, ok in (
            ("config mode is 'live'", self.config_mode_is_live),
            (f"{C.LIVE_ENV_VAR} set correctly", self.environment_variable_set),
            (f"{C.LIVE_ACK_FILENAME} present and valid", self.acknowledgment_file_valid),
        ):
            lines.append(f"  [{'x' if ok else ' '}] {label}")
        if self.failures:
            lines.append("")
            lines.extend(f"  - {f}" for f in self.failures)
        return "\n".join(lines)


def check_live_arming(mode: RunMode, *, root: str | Path = ".") -> LiveArmingCheck:
    """Evaluate all three locks. Never raises."""
    check = LiveArmingCheck()

    check.config_mode_is_live = mode is RunMode.LIVE
    if not check.config_mode_is_live:
        check.failures.append(
            f"config mode is '{mode.value}', not 'live' (this is the safe default)"
        )

    env_value = os.environ.get(C.LIVE_ENV_VAR, "")
    check.environment_variable_set = env_value == C.LIVE_ENV_VALUE
    if not check.environment_variable_set:
        check.failures.append(
            f"environment variable {C.LIVE_ENV_VAR} must be set to "
            f"'{C.LIVE_ENV_VALUE}'" + (f" (found '{env_value}')" if env_value else "")
        )

    ack_path = Path(root) / C.LIVE_ACK_FILENAME
    if not ack_path.exists():
        check.failures.append(
            f"acknowledgment file {ack_path} does not exist. Create it containing "
            f"exactly: {C.LIVE_ACK_REQUIRED_TEXT}"
        )
    else:
        try:
            content = ack_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            content = ""
            check.failures.append(f"cannot read {ack_path}: {exc}")
        check.acknowledgment_file_valid = content == C.LIVE_ACK_REQUIRED_TEXT
        if not check.acknowledgment_file_valid and content:
            check.failures.append(
                f"{ack_path} does not contain the required text exactly"
            )

    if check.armed:
        logger.critical(
            "LIVE TRADING IS ARMED. All three locks are satisfied. "
            "Real capital is at risk from this point."
        )
    return check


def write_acknowledgment(root: str | Path = ".") -> Path:
    """
    Write the acknowledgment file.

    Exposed as a CLI command rather than called automatically anywhere -- the
    point of the file is that a person creates it on purpose.
    """
    path = Path(root) / C.LIVE_ACK_FILENAME
    path.write_text(C.LIVE_ACK_REQUIRED_TEXT + "\n", encoding="utf-8")
    logger.warning("wrote live-trading acknowledgment to %s", path)
    return path
