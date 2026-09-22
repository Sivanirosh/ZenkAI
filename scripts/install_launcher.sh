#!/usr/bin/env bash
# Install a desktop entry for ZenkAI so it can be launched from
# the applications menu / dock like any other app.
#
# Re-run this script if you move the repo to a different path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV_SCRIPT="$REPO_ROOT/scripts/dev.sh"
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_FILE="$APPS_DIR/zenkai.desktop"

if [ ! -x "$DEV_SCRIPT" ]; then
  echo "Error: $DEV_SCRIPT is not executable." >&2
  echo "       Run: chmod +x $DEV_SCRIPT" >&2
  exit 1
fi

# An inline SVG icon (book + spark) kept inside the repo so the launcher
# has something nicer than the generic cog. Written on every install so
# the source of truth stays in one place.
ICON_DIR="$REPO_ROOT/assets"
ICON_PATH="$ICON_DIR/zenkai.svg"
mkdir -p "$ICON_DIR"
cat > "$ICON_PATH" <<'SVG'
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#c7a572"/>
      <stop offset="100%" stop-color="#8a6a3b"/>
    </linearGradient>
  </defs>
  <rect x="10" y="14" width="108" height="100" rx="12" fill="url(#g)"/>
  <rect x="22" y="26" width="84" height="76" rx="4" fill="#fdf6e9"/>
  <path d="M64 28 v74" stroke="#8a6a3b" stroke-width="2"/>
  <g fill="#4a3a22" font-family="Georgia, serif" text-anchor="middle">
    <text x="43" y="62" font-size="22" font-style="italic">L</text>
    <text x="85" y="62" font-size="22" font-style="italic">M</text>
  </g>
  <g fill="#c7952b">
    <circle cx="64" cy="82" r="3.2"/>
    <path d="M64 74 l2 6 6 2 -6 2 -2 6 -2 -6 -6 -2 6 -2 z" opacity="0.7"/>
  </g>
</svg>
SVG

mkdir -p "$APPS_DIR"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=ZenkAI
GenericName=German reader dev server
Comment=Start the ZenkAI backend + frontend and open the reader
Exec=$DEV_SCRIPT
Path=$REPO_ROOT
Icon=$ICON_PATH
Terminal=true
Categories=Education;
StartupNotify=true
Keywords=german;reader;fastapi;ollama;
EOF

chmod 644 "$DESKTOP_FILE"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPS_DIR" 2>/dev/null || true
fi

echo "Installed launcher:"
echo "  $DESKTOP_FILE"
echo
echo "You can now find 'ZenkAI' in your application menu / search."
echo "To remove:  rm '$DESKTOP_FILE'"
