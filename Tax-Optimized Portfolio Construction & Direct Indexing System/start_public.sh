#!/bin/bash
#===============================================================================
#  TAX-OPTIMIZED PORTFOLIO DASHBOARD - PUBLIC ACCESS LAUNCHER
#===============================================================================

echo ""
echo "============================================================"
echo "🚀 TAX-OPTIMIZED PORTFOLIO DASHBOARD - PUBLIC ACCESS"
echo "============================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

cd "$(dirname "$0")"

# Check if dashboard is already running
if lsof -Pi :8050 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Dashboard already running on port 8050${NC}"
else
    echo -e "${BLUE}📊 Starting dashboard server...${NC}"
    python dashboard.py --public &
    DASH_PID=$!
    sleep 3
    echo -e "${GREEN}✓ Dashboard started (PID: $DASH_PID)${NC}"
fi

echo ""
echo -e "${BLUE}🌐 Creating Cloudflare tunnel (no password required)...${NC}"
echo ""

# Kill any existing tunnel
pkill -f "cloudflared tunnel" 2>/dev/null
sleep 1

# Start cloudflared tunnel and capture URL
cloudflared tunnel --url http://localhost:8050 2>&1 | while read line; do
    echo "$line"
    # Extract and display the public URL
    if echo "$line" | grep -q "trycloudflare.com"; then
        URL=$(echo "$line" | grep -oE 'https://[a-z-]+\.trycloudflare\.com')
        if [ ! -z "$URL" ]; then
            echo ""
            echo "============================================================"
            echo -e "${GREEN}✅ PUBLIC ACCESS ENABLED${NC}"
            echo "============================================================"
            echo ""
            echo -e "   ${CYAN}🌐 PUBLIC URL:${NC}  $URL"
            echo -e "   ${BLUE}📍 LOCAL URL:${NC}   http://localhost:8050"
            echo ""
            echo "   📱 Share the public URL to access from:"
            echo "      • Any laptop/desktop"
            echo "      • Mobile phones (iPhone, Android)"
            echo "      • Any browser worldwide"
            echo ""
            echo "   ✨ NO PASSWORD REQUIRED - Direct access!"
            echo ""
            echo "   Press Ctrl+C to stop"
            echo ""
        fi
    fi
done
