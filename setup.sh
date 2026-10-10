#!/bin/bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON:-python3}"

echo "Setting up Gemini Web Proxy for OpenCode..."
echo "=========================================="

if ! "$python_bin" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

echo "✓ Python version check passed: $("$python_bin" --version)"

# Install Python dependencies
echo "Installing Python dependencies..."
"$python_bin" -m pip install -r requirements.txt

# Install Playwright browsers
echo "Installing Playwright browsers..."
"$python_bin" -m playwright install chromium

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: $python_bin run.py"
echo "2. Log in to your Google account when browser opens"
echo "3. Configure OpenCode with the provider settings from README.md"
echo ""
echo "For detailed instructions, see README.md"
