"""The tool catalog and dispatch registry from SPEC.md Part 4.

Kept small and safe for a first version: no filesystem or network access,
nothing that needs its own confirmation prompt. calculator() is adapted from
joblib's own eval_expr (joblib/joblib/joblib/_utils.py, itself credited to
https://stackoverflow.com/a/9558001), narrowed to this project's operator
set only (+ - * / ** and unary minus, dropping joblib's extra floor division
and modulo) and never real eval(), since a from scratch small model's output
cannot be trusted as safe input to eval.
"""
import ast
import datetime
import functools
import operator as op

_MAX_EXPR_LENGTH = 60
_MAX_MAGNITUDE = 10 ** 6
# _limit below can only reject a result it has already paid for, and ** is
# the one operator that can make paying for it expensive: every operand is
# itself bounded to _MAX_MAGNITUDE, but 999999**999999 is still a ~6 million
# digit integer that measured 2.03s to build and then could not even be
# formatted into _limit's own error message (CPython's 4300 digit int -> str
# cap), leaking "Exceeds the limit (4300 digits)..." to the user instead of
# "result ... is too large". So the exponent is bounded before the power is
# computed. This is the operand level guard the StackOverflow answer this
# module is adapted from pairs with its limit() decorator, and the same one
# asteval enforces as a max exponent; only limit() was carried over here.
# 100 is that answer's own bound and is generous at this magnitude: the
# largest power it admits, 1000000**100, is 601 digits and builds in under a
# microsecond.
_MAX_EXPONENT = 100


def _power(base, exponent):
    """op.pow, with the exponent bounded before the power is computed.

    Only int ** positive int can be expensive: a negative int exponent goes
    through float pow and a float base overflows, both in constant time and
    both measured, so neither is rejected here.
    """
    if isinstance(base, int) and isinstance(exponent, int) and exponent > _MAX_EXPONENT:
        raise ValueError(f"exponent {exponent} is too large, max is {_MAX_EXPONENT}")
    return op.pow(base, exponent)


_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: _power,
    ast.USub: op.neg,
}


def _limit(max_):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            try:
                magnitude = abs(result)
            except TypeError:
                pass
            else:
                if magnitude > max_:
                    raise ValueError(f"result {result} is too large, max is {max_}")
            return result
        return wrapper
    return decorator


@_limit(_MAX_MAGNITUDE)
def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    elif isinstance(node, ast.BinOp):
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    elif isinstance(node, ast.UnaryOp):
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    else:
        raise TypeError(node)


def calculator(expression):
    """Arithmetic only: numbers and + - * / ** ( ), never real eval()."""
    if len(expression) > _MAX_EXPR_LENGTH:
        raise ValueError(
            f"expression {expression[:_MAX_EXPR_LENGTH]!r}... is too long, "
            f"max length is {_MAX_EXPR_LENGTH}, got {len(expression)}"
        )
    try:
        return _eval_node(ast.parse(expression, mode="eval").body)
    except (TypeError, SyntaxError, OverflowError, KeyError, ZeroDivisionError) as e:
        raise ValueError(f"{expression!r} is not a valid or supported arithmetic expression") from e


def current_datetime():
    """No arguments. Returns the current date and time."""
    return datetime.datetime.now().isoformat(timespec="seconds")


TOOLS = {
    "calculator": calculator,
    "current_datetime": current_datetime,
}
