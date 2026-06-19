#!/bin/bash
# run_training.sh — Phase 4 launcher (delegates to adaptive Python orchestrator)
#
# The actual curriculum logic lives in ai/run_curriculum.py which:
#   - Trains each level in step blocks (not fixed budgets)
#   - Evaluates after every block
#   - Only advances when the model is MASTERED (reward + completion thresholds met
#     AND reward has plateaued — i.e., no longer improving)
#   - If still improving → lets it bake longer (up to per-level max budget)
#   - Saves best_model checkpoint after every block that beats previous best
#   - Logs curriculum state to ai/models/Phase3/curriculum_state.json
#
# Usage:
#   ./run_training.sh                        # full L1→L6
#   ./run_training.sh --start-level 4        # resume from L4
#   ./run_training.sh --start-level 4 \
#       --load ai/models/Phase3/L3_7Trains_Best/best_model.zip

set -e
cd "$(dirname "$0")"
source venv/bin/activate

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🚂  CSMT-Manmad Adaptive Curriculum — Phase 4               ║"
echo "║  Trains: 2 → 5 → 7 → 10 → 15 → 25                          ║"
echo "║  Advances only when model is mastered, not just by steps     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

python ai/run_curriculum.py "$@"
