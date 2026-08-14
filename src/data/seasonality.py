"""Frequency-alias parsing and seasonal periods for the GiftEvalPretrain corpus."""

from pandas.tseries.frequencies import to_offset

# GiftEvalPretrain stores gluonts/old-pandas frequency aliases (e.g. "M",
# "5T", "A-DEC") that newer pandas rejects in favor of "ME"/"min"/"YE-DEC".
_OLD_TO_NEW_FREQ_BASE = {
    "Y": "YE",
    "A": "YE",
    "Q": "QE",
    "M": "ME",
    "H": "h",
    "T": "min",
    "S": "s",
    "U": "us",
}


def parse_offset(freq: str):
    """Parses a GiftEvalPretrain frequency alias into a pandas offset,
    translating legacy aliases and preserving any multiplier and anchor
    (e.g. "5T" -> 5 minutes, "W-SUN" -> weekly)."""
    base, _, anchor = freq.partition("-")
    split = next((i for i, c in enumerate(base) if not c.isdigit()), len(base))
    mult, code = base[:split], base[split:]
    new_code = _OLD_TO_NEW_FREQ_BASE.get(code, code)
    anchor_suffix = f"-{anchor}" if anchor else ""
    return to_offset(f"{mult}{new_code}{anchor_suffix}")


def seasonal_period(freq: str) -> int:
    """Number of steps in the dominant seasonal cycle for a frequency, the
    MASE denominator's lag.

    Derived from the offset's actual duration rather than a lookup table of
    literal alias strings, so multipliers and anchors (e.g. "4S", "30T",
    "W-SUN", "A-DEC") are handled without enumerating every spelling. Follows
    the GIFT-Eval / gluonts `get_seasonality` convention: sub-daily
    frequencies take the daily cycle, daily takes the weekly cycle, weekly and
    yearly have no shorter cycle (1), monthly takes 12 and quarterly 4.
    """
    offset = parse_offset(freq)
    try:
        seconds = offset.nanos / 1e9
    except ValueError:
        # Non-fixed durations (weeks, months, quarters, years).
        name = type(offset).__name__
        if name == "Week":
            return 1
        if name.startswith("Month"):
            return 12
        if name.startswith("Quarter"):
            return 4
        return 1  # yearly and coarser
    if seconds < 86400:
        return round(86400 / seconds)  # sub-daily -> daily cycle
    if seconds == 86400:
        return 7  # daily -> weekly cycle
    return 1
