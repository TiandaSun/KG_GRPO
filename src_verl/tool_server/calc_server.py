"""v21 W23 — Calculator tool-server (the matched NON-KG interface control).

Mirrors the KG server's HTTP contract (POST /retrieve, GET /health) but the tool
is a safe arithmetic evaluator instead of a graph query. The CRITICAL property
for W23: tool FAILURES LEAK PRETRAINING-ALIGNED NL SIGNAL (Python-style error
messages: "ERROR: division by zero", "ERROR: unknown name 'x'", "ERROR: malformed
expression"), in contrast to the KG interface whose failures are a silent `[]`.

This isolates the paper's thesis variable: same GRPO recipe + same self-verifiable
retrieval reward, swap ONLY the tool interface (KG-symbolic-silent → calc-NL-verbose).

Safe evaluation: AST-walk allowing only numeric literals + - * / // % ** and
parentheses/unary-minus. No names, calls, attributes, comprehensions, etc.

Run:
    python -m src_verl.tool_server.calc_server --port 18950
Request body (RetrieveRequest-compatible):
    {"action": "calculate", "entity": "<expression>", "relation": null}
Response:
    {"action": ..., "entity": ..., "results": ["<value>"] or ["ERROR: ..."], "found": bool}
"""
from __future__ import annotations

import argparse
import ast
import logging
import operator as op
import re
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Allowed binary / unary operators (arithmetic only).
_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


class CalcError(Exception):
    """Raised with a pretraining-aligned NL message on any eval failure."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError(f"unsupported constant {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise CalcError(f"unsupported operator {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        # ** guard against huge exponents (DoS / overflow).
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise CalcError("exponent too large")
        try:
            return _BIN_OPS[type(node.op)](left, right)
        except ZeroDivisionError:
            raise CalcError("division by zero")
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise CalcError(f"unsupported unary operator {type(node.op).__name__}")
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name):
        raise CalcError(f"unknown name '{node.id}' (only numeric expressions allowed)")
    if isinstance(node, ast.Call):
        fn = getattr(node.func, "id", "?")
        raise CalcError(f"function calls not allowed ('{fn}')")
    raise CalcError(f"malformed expression near {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    """Evaluate an arithmetic expression. Return the numeric result as a string,
    or an "ERROR: <nl message>" string on any failure (the NL-signal property)."""
    expr = (expression or "").strip()
    if not expr:
        return "ERROR: empty expression"
    # Strip a leading function-name wrapper like "calculate(...)" / "compute(...)"
    # ONLY when an identifier precedes the opening paren — must NOT mangle a bare
    # parenthesised expression such as "1/(2-2)".
    m = re.match(r"^[A-Za-z_]\w*\((.*)\)$", expr, re.DOTALL)
    if m and m.group(1).strip():
        expr = m.group(1).strip()
    expr = expr.lstrip("=").strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"ERROR: malformed expression ({e.msg})"
    try:
        val = _eval_node(tree)
    except CalcError as e:
        return f"ERROR: {e}"
    except OverflowError:
        return "ERROR: numeric overflow"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    # Normalise int-valued floats to int string (so "6.0" reads as "6").
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val)


# --- HTTP layer (KG-server-compatible contract) ---------------------------
class RetrieveRequest(BaseModel):
    action: str = "calculate"
    entity: str  # the expression (named 'entity' to match the KG tool contract)
    relation: str | None = None


class RetrieveResponse(BaseModel):
    action: str
    entity: str
    relation: str | None = None
    results: list[str]
    found: bool


class HealthResponse(BaseModel):
    status: str
    tool: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Calculator tool-server up (safe arithmetic, NL-error failures).")
    yield


app = FastAPI(title="Calculator Tool Server", version="1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", tool="calculator")


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    result = safe_calculate(req.entity)
    return RetrieveResponse(
        action=req.action, entity=req.entity, relation=req.relation,
        results=[result], found=not result.startswith("ERROR"),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description="Calculator tool server (W23 non-KG control)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=18950)
    p.add_argument("--log-level", default="warning")
    # quick self-test path (no server) for CI / validation
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        cases = ["17*23", "(17*23)+456", "100/0", "2**500", "foo+1",
                 "calculate(12*12)", "= 3 + 4", "", "1/(2-2)", "10 // 3"]
        for c in cases:
            print(f"{c!r:24s} -> {safe_calculate(c)!r}")
        return
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
