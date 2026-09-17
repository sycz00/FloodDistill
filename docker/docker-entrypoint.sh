#!/usr/bin/env bash
# Arguments starting with a dash go to the experiment entry point; anything else is run
# as-is, so the underlying CLIs stay reachable:
#   docker run flooddistill --mode all --dataset s11
#   docker run flooddistill python scripts/train.py ofa --help
set -e

# With --user $(id -u):$(id -g) the uid has no /etc/passwd entry, so getpass.getuser()
# (wandb, matplotlib) raises KeyError and $HOME still points at the root-owned /root.
# Both are set here so --user works without extra setup.
if [ "$(id -u)" != "0" ]; then
    export USER="${USER:-uid$(id -u)}"
    export LOGNAME="$USER"
    if [ ! -w "${HOME:-/root}" ]; then
        export HOME=/tmp
    fi
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
    if [ ! -w "$HF_HOME" ] 2>/dev/null; then
        export HF_HOME="$HOME/.cache/huggingface"
    fi
    export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"
    mkdir -p "$HF_HOME" "$MPLCONFIGDIR" 2>/dev/null || true
fi

if [ "$#" -eq 0 ]; then
    set -- --help
fi

case "$1" in
    -*) exec conda run --no-capture-output -n esi python docker/entrypoint.py "$@" ;;
     *) exec conda run --no-capture-output -n esi "$@" ;;
esac
