"""Exact decimal-second coverage, without epsilon, clipping or retiming.

The author returns decimal seconds after its millisecond FIX vector. Turning
that decimal back into binary float * sample_rate can invent a sample overlap
at an exactly shared boundary. v2 uses the preserved decimal value as a rational
number. Legacy v1 remains available only to inspect its historical coordinates.
"""
from fractions import Fraction
import math

LEGACY_MAPPING = "source_offset_floor_start_ceil_end_v1"
TIME_MAPPING = "source_offset_decimal_floor_start_ceil_end_v2"
SUPPORTED_MAPPINGS = (LEGACY_MAPPING, TIME_MAPPING)


def sample_bounds(start_seconds, end_seconds, sample_rate, mapping=TIME_MAPPING):
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("positive integer source sample rate required")
    if mapping not in SUPPORTED_MAPPINGS:
        raise ValueError("unknown source clock mapping")
    for value in (start_seconds, end_seconds):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("finite numeric native seconds required")
    if mapping == LEGACY_MAPPING:
        return math.floor(start_seconds * sample_rate), math.ceil(end_seconds * sample_rate)
    return (math.floor(Fraction(str(start_seconds)) * sample_rate),
            math.ceil(Fraction(str(end_seconds)) * sample_rate))
