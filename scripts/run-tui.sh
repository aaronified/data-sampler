#!/usr/bin/env sh
# Launch the data-sampler TUI.
# Works on any Linux distro and macOS (POSIX sh — no bashisms).
# Usage: ./run-tui.sh [file.csv] [--sheet NAME]

WHEEL_URL="https://github.com/aaronified/data-sampler/releases/download/v3.0.1/data_sampler-3.0.1-py3-none-any.whl"

if command -v data-sampler >/dev/null 2>&1; then
    exec data-sampler --tui "$@"
fi

for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import data_sampler" >/dev/null 2>&1; then
        exec "$py" -m data_sampler --tui "$@"
    fi
done

echo "data-sampler is not installed. Install it with:" >&2
echo "  pip install $WHEEL_URL" >&2
exit 1
