#!/bin/sh
# Run a command inside a no-network bubblewrap sandbox that sees ONLY: system dirs (read-only),
# /etc (read-only), and one arena copy, mounted at the arena's ORIGINAL absolute path (so any
# hard-coded paths in Opus's code still work). Its data/train and .venv are mounted read-only
# (they are hard links to the original arena). $HOME, the project, data/eval_frozen, other arenas
# and everything else on disk are absent. Memory/CPU capped like the fair-fight runs.
#   final_bwrap.sh <arena_copy_dir> <arena_original_abs_path> <timeout> <cmd> [args...]
set -eu
COPY=$1; ABS=$2; TO=$3; shift 3
[ -d "$COPY" ] && [ -n "$ABS" ] || { echo "bad args" >&2; exit 2; }
exec systemd-run --user --scope --quiet -p MemoryMax=3G -p OOMPolicy=continue -p CPUQuota=350% \
  timeout -k 60 "$TO" \
  bwrap --unshare-all --die-with-parent --new-session \
    --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/sbin /sbin \
    --ro-bind /etc /etc --proc /proc --dev /dev --tmpfs /tmp \
    --bind "$COPY" "$ABS" --ro-bind "$COPY/data/train" "$ABS/data/train" --ro-bind "$COPY/.venv" "$ABS/.venv" \
    --chdir "$ABS" --clearenv --setenv PATH /usr/bin:/bin --setenv HOME "$ABS" --setenv TMPDIR /tmp \
    --setenv OMP_NUM_THREADS 3 --setenv MKL_NUM_THREADS 3 --setenv OPENBLAS_NUM_THREADS 3 --setenv NUMEXPR_NUM_THREADS 3 \
    "$@"
