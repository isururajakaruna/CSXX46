#!/bin/bash

# DQN Inference Runner
# Interactive script to run trained DQN models for trading inference

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Detect project root (go up 2 levels: custom_scripts -> custom_modules -> ats)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}    DQN Trading Model - Inference Mode${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if ATS server is running
echo -e "${BLUE}→ Checking ATS server...${NC}"
if ! curl -s http://localhost:5010/health > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ ATS server not running on port 5010${NC}"
    echo -e "${YELLOW}  Please start it first:${NC}"
    echo -e "  ${GREEN}cd $PROJECT_ROOT && ./start.sh${NC}"
    exit 1
fi
echo -e "${GREEN}✓ ATS server is running${NC}"
echo ""

# Find available trained models
MODELS_DIR="$PROJECT_ROOT/custom_modules/strategies/DQN/saved_models"
echo -e "${BLUE}→ Scanning for trained models...${NC}"

if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${RED}✗ No saved_models directory found${NC}"
    echo -e "  Please train a model first using: ${GREEN}./train_interactive_DQN.sh${NC}"
    exit 1
fi

# List all .pth files
mapfile -t PTH_FILES < <(find "$MODELS_DIR" -name "*.pth" -type f | sort -r)

if [ ${#PTH_FILES[@]} -eq 0 ]; then
    echo -e "${RED}✗ No trained models found${NC}"
    echo -e "  Please train a model first using: ${GREEN}./train_interactive_DQN.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found ${#PTH_FILES[@]} model(s)${NC}"
echo ""

# Display available models
echo -e "${CYAN}Available Trained Models:${NC}"
echo ""
for i in "${!PTH_FILES[@]}"; do
    MODEL_PATH="${PTH_FILES[$i]}"
    RELATIVE_PATH="${MODEL_PATH#$PROJECT_ROOT/}"
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    MODEL_DATE=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$MODEL_PATH" 2>/dev/null || stat -c "%y" "$MODEL_PATH" 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1)
    echo -e "  ${YELLOW}[$((i+1))]${NC} $RELATIVE_PATH"
    echo -e "      Size: $MODEL_SIZE | Modified: $MODEL_DATE"
    echo ""
done

# Select model
while true; do
    read -p "$(echo -e ${YELLOW}"Select model [1-${#PTH_FILES[@]}]: "${NC})" MODEL_CHOICE
    if [[ "$MODEL_CHOICE" =~ ^[0-9]+$ ]] && [ "$MODEL_CHOICE" -ge 1 ] && [ "$MODEL_CHOICE" -le ${#PTH_FILES[@]} ]; then
        SELECTED_MODEL="${PTH_FILES[$((MODEL_CHOICE-1))]}"
        SELECTED_MODEL_RELATIVE="${SELECTED_MODEL#$PROJECT_ROOT/}"
        break
    else
        echo -e "${RED}Invalid choice. Please enter a number between 1 and ${#PTH_FILES[@]}${NC}"
    fi
done

echo ""
echo -e "${GREEN}✓ Selected: $SELECTED_MODEL_RELATIVE${NC}"
echo ""

# Select data source
CANDIDATE_DIRS=(
    "$PROJECT_ROOT/ats/data"
    "$PROJECT_ROOT/data"
    "$PROJECT_ROOT/custom_modules/data"
)

DATA_FILES=()
for dir in "${CANDIDATE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        while IFS= read -r -d '' file; do
            DATA_FILES+=("$file")
        done < <(find "$dir" -name "*.csv" -type f -print0 2>/dev/null)
    fi
done

if [ ${#DATA_FILES[@]} -eq 0 ]; then
    echo -e "${RED}✗ No data files found${NC}"
    exit 1
fi

echo -e "${CYAN}Available Data Files:${NC}"
echo ""
for i in "${!DATA_FILES[@]}"; do
    FILE_PATH="${DATA_FILES[$i]}"
    RELATIVE_PATH="${FILE_PATH#$PROJECT_ROOT/}"
    FILE_SIZE=$(du -h "$FILE_PATH" | cut -f1)
    LINE_COUNT=$(wc -l < "$FILE_PATH" | tr -d ' ')
    echo -e "  ${YELLOW}[$((i+1))]${NC} $RELATIVE_PATH"
    echo -e "      Size: $FILE_SIZE | Rows: $LINE_COUNT"
    echo ""
done

while true; do
    read -p "$(echo -e ${YELLOW}"Select data file [1-${#DATA_FILES[@]}]: "${NC})" DATA_CHOICE
    if [[ "$DATA_CHOICE" =~ ^[0-9]+$ ]] && [ "$DATA_CHOICE" -ge 1 ] && [ "$DATA_CHOICE" -le ${#DATA_FILES[@]} ]; then
        SELECTED_DATA="${DATA_FILES[$((DATA_CHOICE-1))]}"
        SELECTED_DATA_RELATIVE="${SELECTED_DATA#$PROJECT_ROOT/}"
        break
    else
        echo -e "${RED}Invalid choice. Please enter a number between 1 and ${#DATA_FILES[@]}${NC}"
    fi
done

echo ""
echo -e "${GREEN}✓ Selected: $SELECTED_DATA_RELATIVE${NC}"
echo ""

# Trading parameters
read -p "$(echo -e ${YELLOW}"Order value % (default: 10): "${NC})" ORDER_VALUE_PCT
ORDER_VALUE_PCT=${ORDER_VALUE_PCT:-10}

read -p "$(echo -e ${YELLOW}"Initial quote balance (default: 10000): "${NC})" INITIAL_BALANCE
INITIAL_BALANCE=${INITIAL_BALANCE:-10000}

# Epsilon for exploration
read -p "$(echo -e ${YELLOW}"Epsilon (0.0=greedy, default: 0.0): "${NC})" EPSILON
EPSILON=${EPSILON:-0.0}

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}    Configuration Summary${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${BLUE}Model:${NC}             $SELECTED_MODEL_RELATIVE"
echo -e "  ${BLUE}Data:${NC}              $SELECTED_DATA_RELATIVE"
echo -e "  ${BLUE}Initial Balance:${NC}   $INITIAL_BALANCE USDT"
echo -e "  ${BLUE}Order Value %:${NC}     $ORDER_VALUE_PCT%"
echo -e "  ${BLUE}Epsilon:${NC}           $EPSILON"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Generate job name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="DQN_Inference_${TIMESTAMP}"

# Create JSON config
CONFIG_FILE="$PROJECT_ROOT/custom_modules/strategies/DQN/inference_${TIMESTAMP}.json"

cat > "$CONFIG_FILE" << EOF
{
  "job_name": "$JOB_NAME",
  "strategy": "DQN",
  "base_symbol": "BTC",
  "quote_symbol": "USDT",
  "data_source": "$SELECTED_DATA_RELATIVE",
  "initial_quote_balance": $INITIAL_BALANCE,
  "initial_base_balance": 0,
  
  "strategy_config": {
    "training_mode": false,
    "model_path": "$SELECTED_MODEL_RELATIVE",
    "epsilon": $EPSILON,
    
    "order_value_pct": $ORDER_VALUE_PCT,
    "max_position_size": 1.0,
    "log_trade_activity": true,
    
    "use_indicators": true,
    "ma_window": 20,
    "rsi_window": 14,
    
    "reward_config": {
      "use_reward": true,
      "reward_type": "pnl_pct",
      "penalty_hold": 0.0,
      "penalty_flip": 0.0,
      "reward_scale": 1.0
    }
  }
}
EOF

echo -e "${GREEN}✓ Config saved: $CONFIG_FILE${NC}"
echo ""

# Submit job
echo -e "${BLUE}→ Submitting inference job to ATS...${NC}"
RESPONSE=$(curl -s -X POST http://localhost:5010/v1/jobs \
  -H "Content-Type: application/json" \
  -d @"$CONFIG_FILE")

# Parse job ID from response
JOB_ID=$(echo "$RESPONSE" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$JOB_ID" ]; then
    echo -e "${RED}✗ Failed to submit job${NC}"
    echo -e "Response: $RESPONSE"
    exit 1
fi

echo -e "${GREEN}✓ Job submitted successfully!${NC}"
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}    Inference Job Details${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${BLUE}Job ID:${NC}    ${GREEN}$JOB_ID${NC}"
echo -e "  ${BLUE}Job Name:${NC}  $JOB_NAME"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Useful Commands:${NC}"
echo ""
echo -e "  ${BLUE}# Check job status${NC}"
echo -e "  curl http://localhost:5010/v1/jobs/$JOB_ID"
echo ""
echo -e "  ${BLUE}# Get final results${NC}"
echo -e "  curl http://localhost:5010/v1/jobs/$JOB_ID/results"
echo ""
echo -e "  ${BLUE}# List all jobs${NC}"
echo -e "  curl http://localhost:5010/v1/jobs"
echo ""
echo -e "${GREEN}✓ Inference job started!${NC}"
echo -e "  Monitor progress in ATS logs or use commands above"
echo ""

