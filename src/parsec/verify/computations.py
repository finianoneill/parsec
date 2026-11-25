"""Derived figures as checkable computations (WS3).

A premise holds only what a source says. A figure the source does not state
— a unit conversion, a sum, a ratio — is a Finding with edge_type "computes"
and a typed operation, and the harness recomputes it. Before this, any
`transform_note` on a premise exempted every number in it from the
exact-match check, at no credence cost: the one free bypass of the floor.

An operation names its operands as numbers taken FROM RECORDED PREMISES
(premise id + the value as that premise states it), so a "verified"
computation cannot start from a number no source gave. Arithmetic is exact
decimal — no floats — so a result is the same on every machine and replay.

Three outcomes:
  verified    the harness recomputed the output at the stated precision
  unverified  the operation is free text ("other"): recorded, never
              trusted — capped below the flag threshold, rendered as an
              unverified derived figure, and not evidence for a number
  rejected    the arithmetic does not hold, or an operand is not in its
              premise; returned to the subagent with the expected value
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext

from parsec.verify.containment import canonical_number, sentence_numbers, unsupported_numbers

VERIFIED, UNVERIFIED = "verified", "unverified"

# Factor to the dimension's base unit. Exact by definition (international
# yard and pound, US liquid gallon); temperature is affine and not here.
_UNITS: dict[str, tuple[str, Decimal]] = {
    # length -> metres
    "mm": ("length", Decimal("0.001")),
    "cm": ("length", Decimal("0.01")),
    "m": ("length", Decimal("1")),
    "km": ("length", Decimal("1000")),
    "in": ("length", Decimal("0.0254")),
    "ft": ("length", Decimal("0.3048")),
    "yd": ("length", Decimal("0.9144")),
    "mi": ("length", Decimal("1609.344")),
    # mass -> grams
    "mg": ("mass", Decimal("0.001")),
    "g": ("mass", Decimal("1")),
    "kg": ("mass", Decimal("1000")),
    "t": ("mass", Decimal("1000000")),
    "oz": ("mass", Decimal("28.349523125")),
    "lb": ("mass", Decimal("453.59237")),
    # volume -> litres
    "ml": ("volume", Decimal("0.001")),
    "l": ("volume", Decimal("1")),
    "m3": ("volume", Decimal("1000")),
    "gal": ("volume", Decimal("3.785411784")),
    # time -> seconds
    "ms": ("time", Decimal("0.001")),
    "s": ("time", Decimal("1")),
    "min": ("time", Decimal("60")),
    "h": ("time", Decimal("3600")),
    "day": ("time", Decimal("86400")),
    "week": ("time", Decimal("604800")),
}
_ALIASES = {
    "millimeter": "mm", "millimetre": "mm", "centimeter": "cm", "centimetre": "cm",
    "meter": "m", "metre": "m", "kilometer": "km", "kilometre": "km",
    "inch": "in", "inches": "in", "foot": "ft", "feet": "ft", "yard": "yd", "mile": "mi",
    "milligram": "mg", "gram": "g", "kilogram": "kg", "tonne": "t", "ounce": "oz",
    "pound": "lb", "lbs": "lb",
    "milliliter": "ml", "millilitre": "ml", "liter": "l", "litre": "l",
    "cubic meter": "m3", "cubic metre": "m3", "gallon": "gal",
    "millisecond": "ms", "second": "s", "sec": "s", "minute": "min", "hour": "h",
    "hr": "h", "days": "day", "d": "day", "weeks": "week",
}
SUPPORTED_UNITS = tuple(sorted(_UNITS))

# op -> how many operands it takes (None = two or more)
_ARITY: dict[str, int | None] = {
    "convert": 1, "round": 1, "diff": 2, "ratio": 2, "percent_change": 2, "sum": None,
}
RECOMPUTABLE_OPS = tuple(sorted(_ARITY))


def _unit(name: str | None) -> tuple[str, Decimal] | None:
    if not name:
        return None
    key = " ".join(name.lower().split())
    key = _ALIASES.get(key, key)
    if key not in _UNITS and key.endswith("s"):
        key = _ALIASES.get(key[:-1], key[:-1])
    return _UNITS.get(key)


def _decimal(text: object) -> Decimal | None:
    try:
        value = Decimal(str(text).replace(",", "").replace("−", "-").strip())
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def evidence_text(payload: dict) -> str:
    """The text of a premise or finding AS EVIDENCE for a number or quote.
    An unverified derivation is recorded and shown, but it is evidence for
    nothing: a sentence stating its figure fails the sentence check and is
    marked, which is how the reader is told the figure was not checked."""
    if payload.get("status") == UNVERIFIED:
        return ""
    return str(payload.get("text", ""))


@dataclass
class ComputationCheck:
    status: str | None = None           # VERIFIED / UNVERIFIED; None when rejected
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _show(value: Decimal, precision: Decimal) -> str:
    """A recomputed value with one more digit than the stated precision —
    enough for the subagent to see why its output is off."""
    finer = precision / 10
    with localcontext() as ctx:
        ctx.prec = 40
        return format(value.quantize(finer), "f")


def recompute(operation: dict) -> Decimal | None:
    """The exact result of a recomputable operation, or None when it cannot
    be computed (unknown op, wrong operand count, bad units, division by
    zero). Callers that need the reason use check_operation."""
    values = [_decimal(o.get("value")) for o in operation.get("operands", [])]
    if any(v is None for v in values):
        return None
    op = operation.get("op")
    arity = _ARITY.get(str(op), -1)
    if arity == -1 or (arity is None and len(values) < 2) or (arity and len(values) != arity):
        return None
    numbers = [v for v in values if v is not None]
    with localcontext() as ctx:
        ctx.prec = 40
        if op == "convert":
            src, dst = _unit(operation.get("unit_from")), _unit(operation.get("unit_to"))
            if src is None or dst is None or src[0] != dst[0]:
                return None
            return numbers[0] * src[1] / dst[1]
        if op == "round":
            return numbers[0]
        if op == "sum":
            return sum(numbers, Decimal(0))
        if op == "diff":
            return numbers[0] - numbers[1]
        if numbers[1 if op == "ratio" else 0] == 0:
            return None
        if op == "ratio":
            return numbers[0] / numbers[1]
        return (numbers[1] - numbers[0]) / numbers[0] * 100  # percent_change: from, to


def check_operation(operation: dict, premise_texts: dict[str, str]) -> ComputationCheck:
    """Validate one `computes` operation against the texts of the premises
    its operands name. premise_texts: premise id -> recorded text."""
    result = ComputationCheck()
    op = str(operation.get("op"))
    operands = operation.get("operands") or []
    if not operands:
        result.problems.append("operation has no operands")
        return result

    for i, operand in enumerate(operands):
        pid, value = operand.get("premise_id"), operand.get("value")
        if pid not in premise_texts:
            result.problems.append(
                f"operands[{i}]: {pid} is not one of this finding's premise_ids"
            )
            continue
        number = _decimal(value)
        if number is None:
            result.problems.append(f"operands[{i}]: {value!r} is not a number")
        elif canonical_number(str(value)) not in sentence_numbers(premise_texts[pid]):
            result.problems.append(
                f"operands[{i}]: {value} does not appear in {pid} "
                f"({premise_texts[pid][:120]!r}) — an operand must be a number that premise states"
            )
    output = _decimal(operation.get("output"))
    if output is None:
        result.problems.append(f"output {operation.get('output')!r} is not a number")
    if result.problems:
        return result

    if op == "other":
        result.status = UNVERIFIED  # free text: recorded, never trusted
        return result
    if op not in _ARITY:
        result.problems.append(
            f"unknown op {op!r}; use one of {', '.join(RECOMPUTABLE_OPS)}, or 'other'"
        )
        return result

    arity = _ARITY[op]
    if (arity is None and len(operands) < 2) or (arity is not None and len(operands) != arity):
        need = "two or more operands" if arity is None else f"exactly {arity} operand(s)"
        result.problems.append(f"{op} takes {need}, got {len(operands)}")
        return result
    if op == "convert":
        src, dst = _unit(operation.get("unit_from")), _unit(operation.get("unit_to"))
        if src is None or dst is None:
            unknown = operation.get("unit_from") if src is None else operation.get("unit_to")
            result.problems.append(
                f"unknown unit {unknown!r}; supported: {', '.join(SUPPORTED_UNITS)}"
            )
            return result
        if src[0] != dst[0]:
            result.problems.append(
                f"cannot convert {operation.get('unit_from')} ({src[0]}) "
                f"to {operation.get('unit_to')} ({dst[0]})"
            )
            return result

    precision = _decimal(operation.get("precision"))
    # A power of ten, and the output must sit on that grid: otherwise a
    # coarse "precision" would let any nearby wrong figure through.
    if precision is None or precision <= 0 or precision.normalize().as_tuple().digits != (1,):
        result.problems.append(
            f"precision {operation.get('precision')!r} must be a power of ten such as 1, 0.1 or 0.01"
        )
        return result
    assert output is not None
    if output % precision != 0:
        result.problems.append(f"output {operation.get('output')} is not stated to precision {precision}")
        return result

    computed = recompute(operation)
    if computed is None:
        result.problems.append(f"{op} cannot be computed from these operands (division by zero?)")
        return result
    if abs(computed - output) > precision / 2:
        result.problems.append(_mismatch(operation, computed, precision))
        return result
    result.status = VERIFIED
    return result


def _mismatch(operation: dict, computed: Decimal, precision: Decimal) -> str:
    values = [str(o.get("value")) for o in operation["operands"]]
    shown, said = _show(computed, precision), operation.get("output")
    op = operation["op"]
    if op == "convert":
        return (
            f"{values[0]} {operation.get('unit_from')} is {shown} {operation.get('unit_to')}; "
            f"the finding says {said}"
        )
    how = {
        "sum": " + ".join(values), "diff": " - ".join(values), "ratio": " / ".join(values),
        "round": values[0],
        "percent_change": f"the change from {values[0]} to {values[-1]} in percent",
    }[op]
    return f"{how} is {shown}; the finding says {said}"


def check_finding(
    text: str,
    edge_type: str,
    operation: dict | None,
    premise_texts: dict[str, str],
) -> ComputationCheck:
    """Everything the harness checks about one finding's numbers.

    - A `computes` finding needs an operation, and the operation must hold.
    - Every number in ANY finding's text must appear in the premises it
      cites, or be the output of its own operation. Without this a figure
      could be laundered through a finding: the sentence check accepts
      numbers found in a cited finding's text.
    """
    if edge_type == "computes":
        if operation is None:
            return ComputationCheck(problems=["a computes finding needs an operation"])
        result = check_operation(operation, premise_texts)
        if not result.ok:
            return result
        evidence = [*premise_texts.values(), str(operation.get("output"))]
    else:
        if operation is not None:
            return ComputationCheck(
                problems=[f"only computes findings carry an operation (this one is {edge_type})"]
            )
        result = ComputationCheck()
        evidence = list(premise_texts.values())
    for number in unsupported_numbers(text, evidence):
        hint = (
            "state the operation's output as given"
            if edge_type == "computes"
            else "if it is derived, submit it as a computes finding with an operation"
        )
        result.problems.append(
            f"number {number!r} in the finding is not in the premises it cites; {hint}"
        )
    if result.problems:
        result.status = None
    return result
