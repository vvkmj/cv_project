#!/bin/bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT_DIR="research/screenshots"
mkdir -p "$OUT_DIR"

"$CHROME" --headless --disable-gpu --user-data-dir=/tmp/c_user_profile --window-size=1280,850 --screenshot="$OUT_DIR/fig_dashboard_overview.png" "http://127.0.0.1:3000"

echo "Captured UI overview screenshot!"

