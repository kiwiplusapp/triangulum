"""
Triangulum -- autonomous multi-venue arbitrage engine.

The short version of how it works:

    market data  ->  currency graph  ->  negative-cycle search  ->  opportunity
                                                                        |
                     ledger  <-  executor  <-  cycle plan  <-  EV gate  +  learner

Every stage is replaceable and every stage is measured. The parts that decide
whether real money moves -- the EV gate, the risk guardrails, and the live
arming lock -- are deliberately the most conservative code in the repository.
"""

from triangulum.version import __version__, VERSION_INFO

__all__ = ["__version__", "VERSION_INFO"]
