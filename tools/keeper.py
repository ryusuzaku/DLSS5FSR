#!/usr/bin/env python3
"""tools/keeper.py -- master re-run of every Stage-1 static gate (gate K).

Runs the full chain in dependency order, fail-closed: each gate must
exit 0 AND show its documented green signal (runbook sections 7, 8,
9/10/11, 12, 13, 14, 16, 17, 18, 19). First failure prints KEEPER-FAIL with
the gate and detail, exit 1. All green prints per-gate PASS lines plus
KEEPER-OK, exit 0.

Keeper output is byte-deterministic (no timings printed). Slowest
gate overall (~20 min total); runbook 15, HANDOFF 33.10.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def check_esum(o):
    assert o.count("axis:NO") == 18, o.count("axis:NO")
    assert "axis:YES" not in o
    return "axisNO=18 axisYES=0"


def check_qk(o):
    assert "TAINT" not in o and "NODEF" not in o and "WALK-ASSERT" not in o
    assert "totals: loads=768 taints=0 texs=768 cskipped=152" in o
    return "zero-taint"


def check_terminal(want):
    def check(o):
        assert o.splitlines() and o.splitlines()[-1] == want, \
            o.splitlines()[-1:]
        return "terminal"
    return check


def check_fanin(o):
    assert len(o.splitlines()) == 90, len(o.splitlines())
    return "lines=90"


GATES = [
    ("tools/esum_ownership.py", check_esum),
    ("tools/qk_addrs.py", check_qk),
    ("tools/o_pairing.py", check_terminal("PAIRING-OK")),
    ("tools/o_fanin.py", check_fanin),
    ("tools/o_tokens.py", check_terminal("OTOKENS-OK")),
    ("tools/o_vcover.py", check_terminal("VCOVER-OK")),
    ("tools/o_micro.py", check_terminal("MMICRO-OK")),
    ("tools/o_rowmajor.py", check_terminal("ROWMAJOR-OK")),
    ("tools/o_frontend.py", check_terminal("FRONTEND-OK")),
    ("tools/o_halves.py", check_terminal("HALVES-OK")),
    ("tools/o_carve.py", check_terminal("CARVE-OK")),
]


def main():
    for tool, check in GATES:
        try:
            p = subprocess.run([PY, os.path.join(ROOT, tool)],
                               cwd=ROOT, capture_output=True, text=True,
                               timeout=1500)
        except subprocess.TimeoutExpired:
            print("keeper: %s FAIL timeout" % tool)
            print("KEEPER-FAIL")
            return 1
        if p.returncode != 0:
            tail = (p.stderr.strip().splitlines() or [""]) [-1][:160]
            print("keeper: %s FAIL exit=%d %s" %
                  (tool, p.returncode, tail))
            print("KEEPER-FAIL")
            return 1
        try:
            detail = check(p.stdout)
        except AssertionError as e:
            print("keeper: %s FAIL signal %s" % (tool, e))
            print("KEEPER-FAIL")
            return 1
        print("keeper: %s exit=0 %s PASS" % (tool, detail))
    print("KEEPER-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
