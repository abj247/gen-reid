"""Paired significance testing used throughout the repository.

Two rules are enforced here rather than left to each call site, because both
were the source of real errors during development.

Exact McNemar, not chi-square. Comparisons are between two arms answering the
same questions, so the only informative quantity is the discordant pairs, and
at the counts seen here the asymptotic approximation is not safe.

Every null result is reported as a bound, never as an absence. `mde` returns
the smallest difference the comparison could have detected at the stated power,
so a non-significant result is phrased as no difference larger than that value
rather than as no difference.
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import binomtest, norm


@dataclass
class Comparison:
    n: int
    acc_a: float
    acc_b: float
    delta: float
    p: float
    n_discordant: int
    mde: float

    def describe(self, alpha: float = 0.05) -> str:
        if self.p < alpha:
            return f"{self.delta:+.2f} points (p={self.p:.3g}, n={self.n})"
        return (f"no difference larger than {self.mde:.2f} points at 80% power "
                f"(observed {self.delta:+.2f}, p={self.p:.3g}, n={self.n})")


def mde(n: int, n_discordant: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable effect in accuracy points for a paired binary design."""
    if n == 0 or n_discordant == 0:
        return float("nan")
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    psi = n_discordant / n
    return 100.0 * (z_a + z_b) * (psi ** 0.5) / (n ** 0.5)


def compare(correct_a: dict[str, bool], correct_b: dict[str, bool]) -> Comparison | None:
    """Paired exact McNemar over the keys both arms answered.

    Returns None when the arms share no keys, which is the signature of a join
    across the two id spaces and should fail loudly rather than report zero.
    """
    keys = sorted(set(correct_a) & set(correct_b))
    if not keys:
        return None
    a = [bool(correct_a[k]) for k in keys]
    b = [bool(correct_b[k]) for k in keys]
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    disc = only_a + only_b
    p = binomtest(only_a, disc, 0.5).pvalue if disc else 1.0
    n = len(keys)
    return Comparison(
        n=n,
        acc_a=100.0 * sum(a) / n,
        acc_b=100.0 * sum(b) / n,
        delta=100.0 * (sum(a) - sum(b)) / n,
        p=float(p),
        n_discordant=disc,
        mde=mde(n, disc),
    )
