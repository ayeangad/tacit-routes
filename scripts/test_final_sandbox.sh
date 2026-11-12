#!/bin/sh
# Containment self-test for scripts/final_bwrap.sh (the sandbox the final test uses to run Opus's
# code on eval inputs). Builds a throwaway copy of an existing arena (default run 3), then runs
# escape attempts (must FAIL) and normal work (must SUCCEED). Prints PASS/FAIL per check.
#   sh scripts/test_final_sandbox.sh [run]
set -u
RUN=${1:-3}
TR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$TR/opus_arena_run$RUN"
T="$TR/runs/final/opus/sandbox_test"
COPY="$T/arena"
if [ -e "$COPY" ]; then chmod -R u+w "$COPY"; rm -rf -- "$COPY"; fi
mkdir -p "$COPY/data"
cp -a "$SRC/README.md" "$COPY/" 2>/dev/null; cp -al "$SRC/data/train" "$COPY/data/train"; cp -al "$SRC/.venv" "$COPY/.venv"
B="sh $TR/scripts/final_bwrap.sh $COPY $SRC 5m"
fails=0
must_fail() { name=$1; shift; if $B "$@" >/dev/null 2>&1; then echo "FAIL (escaped): $name"; fails=$((fails+1)); else echo "PASS (blocked): $name"; fi; }
must_ok()   { name=$1; shift; if $B "$@" >/dev/null 2>&1; then echo "PASS (works):   $name"; else echo "FAIL (broken):  $name"; fails=$((fails+1)); fi; }
must_fail "read ~/.bashrc"                 /usr/bin/head -c 10 /home/angad/.bashrc
must_fail "list project data/"             /usr/bin/ls "$TR/data"
must_fail "list eval_frozen"               /usr/bin/ls "$TR/data/eval_frozen"
must_fail "read sealed answers"            /usr/bin/head -c 10 "$TR/data/eval_frozen/raw/score/eval_actual_sequences.json"
must_fail "list sibling arena run2"        /usr/bin/ls "$TR/opus_arena_run2"
must_fail "list project root"              /usr/bin/ls "$TR/PROJECT.md"
must_fail "read deepseek key"              /usr/bin/head -c 1 /home/angad/.deepseek_key
must_fail "network: python https"          .venv/bin/python -c "import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)"
must_fail "network: raw socket to 1.1.1.1" .venv/bin/python -c "import socket; socket.create_connection(('1.1.1.1', 53), timeout=5)"
must_fail "write into /usr"                /usr/bin/touch /usr/should_not_exist
must_fail "write into data/train (ro)"     /usr/bin/touch data/train/should_not_exist
must_fail "write into .venv (ro)"          /usr/bin/touch .venv/should_not_exist
must_fail "see host processes"             /bin/sh -c "test \$(ls /proc | grep -c '^[0-9]') -gt 5"
must_ok   "write inside arena copy"        /usr/bin/touch arena_write_ok
must_ok   "python + numpy/torch/ortools"   .venv/bin/python -c "import numpy, torch, ortools; torch.set_num_threads(3)"
must_ok   "read own train data"            /bin/sh -c "ls data/train | head -1 | grep -q npz"
[ -f "$COPY/arena_write_ok" ] && echo "PASS: write landed in the copy, not the original" || { echo "FAIL: write not in copy"; fails=$((fails+1)); }
[ ! -e "$SRC/arena_write_ok" ] && echo "PASS: original arena untouched" || { echo "FAIL: original arena modified"; fails=$((fails+1)); }
ls /home/angad/should_not_exist /usr/should_not_exist "$SRC/data/train/should_not_exist" 2>/dev/null && { echo "FAIL: escape file exists"; fails=$((fails+1)); }
echo "failures: $fails"
exit $fails
