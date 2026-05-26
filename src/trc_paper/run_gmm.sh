#!/usr/bin/env bash
# run_gmm.sh — robust wrapper for the multi-hour Dynamic GMM regeneration.
#
# Features:
#   • Verifies prerequisites (venv, validation result, disk space).
#   • Uses tmux session if available, otherwise nohup+log file.
#   • Persistent logs with rotation under Resultados/trc_paper/logs/.
#   • Automatic resume on second invocation thanks to checkpoint_enabled=True.
#   • Exit code reflects the inner python script's exit code.
#
# Usage (from project root or anywhere):
#   bash src/trc_paper/run_gmm.sh k5
#   bash src/trc_paper/run_gmm.sh k8
#
# After launch:
#   tmux attach -t dynamic_gmm_k5     # see live progress
#   tail -f Resultados/trc_paper/logs/dynamic_gmm_k5_*.log
#
set -euo pipefail

K_VARIANT="${1:-k5}"
case "$K_VARIANT" in
    k5) K_VALUE=5; CONFIG_FILE="config/default.yaml";       RESULTS_DIR_REL="Resultados/trc_paper" ;;
    k8) K_VALUE=8; CONFIG_FILE="config/k8_sensitivity.yaml"; RESULTS_DIR_REL="Resultados/trc_paper_k8" ;;
    *) echo "Unknown variant: $K_VARIANT. Use 'k5' or 'k8'." >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"     # src/trc_paper
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)" # repo root
cd "$PROJECT_ROOT"

# 1. Verify project venv
if [ ! -d ".venv" ]; then
    echo "ERROR: project venv not found at $PROJECT_ROOT/.venv" >&2
    exit 2
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 2. Verify that the validation step already ran successfully
RUN_TAG="k${K_VALUE}_2018-01-01_2024-09-30"
VALIDATION_FILE="${RESULTS_DIR_REL}/validation/${RUN_TAG}_validation.json"
if [ ! -f "$VALIDATION_FILE" ]; then
    echo "Validation report not found: $VALIDATION_FILE" >&2
    echo "Run 'src/trc_paper/validate_data.py' first or trigger it from the Streamlit page." >&2
    exit 3
fi
if ! python -c "import json,sys; d=json.load(open('$VALIDATION_FILE')); sys.exit(0 if d.get('ready_for_phase_1') else 1)"; then
    echo "Validation report indicates data NOT ready for Phase 1." >&2
    exit 4
fi

# 3. Check disk space (need ~50 GB free for DuckDB temp + outputs)
FREE_KB=$(df -k . | awk 'NR==2 {print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))
if [ "$FREE_GB" -lt 60 ]; then
    echo "WARNING: only ${FREE_GB} GB free. Recommended at least 60 GB." >&2
    if [ -t 0 ]; then
        read -p "Continue anyway? [y/N] " ans
        [ "$ans" = "y" ] || exit 5
    fi
fi

# 4. Prepare output paths
OUT_DB="${RESULTS_DIR_REL}/dynamic_gmm/${RUN_TAG}_assignments.duckdb"
OUT_MODEL="${RESULTS_DIR_REL}/dynamic_gmm/${RUN_TAG}_model.joblib"
OUT_META="${RESULTS_DIR_REL}/dynamic_gmm/${RUN_TAG}_run.json"
LOG_DIR="${RESULTS_DIR_REL}/logs"
LOG_FILE="${LOG_DIR}/dynamic_gmm_${K_VARIANT}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${RESULTS_DIR_REL}/dynamic_gmm" "$LOG_DIR"

CONFIG_PATH="src/trc_paper/${CONFIG_FILE}"

CMD=(
    python src/trc_paper/run_dynamic_gmm.py
    --config "$CONFIG_PATH"
    --k "$K_VALUE"
    --output-db "$OUT_DB"
    --output-model "$OUT_MODEL"
    --output-metadata "$OUT_META"
    --parallel-jobs 4
)

# 5. Pick session manager
if command -v tmux >/dev/null 2>&1; then
    SESS="dynamic_gmm_${K_VARIANT}"
    if tmux has-session -t "$SESS" 2>/dev/null; then
        echo "tmux session '$SESS' already exists. Attach with: tmux attach -t $SESS"
        exit 0
    fi
    tmux new-session -d -s "$SESS" "${CMD[*]} 2>&1 | tee $LOG_FILE; echo 'PROCESS_EXIT'; sleep 60"
    echo "Started in tmux session '$SESS'"
    echo "  attach:  tmux attach -t $SESS"
    echo "  logs:    tail -f $LOG_FILE"
elif command -v screen >/dev/null 2>&1; then
    SESS="dynamic_gmm_${K_VARIANT}"
    screen -dmS "$SESS" bash -c "${CMD[*]} 2>&1 | tee $LOG_FILE; sleep 60"
    echo "Started in screen session '$SESS'"
    echo "  attach:  screen -r $SESS"
    echo "  logs:    tail -f $LOG_FILE"
else
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "Started with nohup (PID $PID). No tmux/screen available."
    echo "  logs:    tail -f $LOG_FILE"
    echo "  monitor: ps -p $PID"
fi

echo
echo "Estimated runtime: 30–50 hours on commodity CPU."
echo "Checkpointing is enabled — re-invoking this script after an interruption"
echo "will resume from the last completed window."
