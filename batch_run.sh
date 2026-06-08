#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# batch_run.sh — Multi-run wrapper for petct_coreg_atlas.sh
#
# Processes all subdirectories in a batch folder using the same hotel
# configuration. Keep single-animal and hotel acquisitions in separate
# batch folders and run once per acquisition type.
#
# Usage:
#   ./batch_run.sh --pipeline <path/to/petct_coreg_atlas.sh> \
#                  --config   <path/to/config.yaml>          \
#                  --batch    <batch_dir>                     \
#                  --hotel    {1|3|4}
#
# Each subdirectory in <batch_dir> must contain:
#   - *PET*.dcm and *CT*.dcm files
#   - SUV file(s): SUV.txt (hotel=1) or SUV_1.txt...SUV_N.txt (hotel>1)
#
# Output per run:   <subdir>/output/
# Per-run log:      <subdir>/output/pipeline.log
# Consolidated QC:  <batch_dir>/batch_QC_summary.csv
# ==============================================================================

PIPELINE=""
CONFIG=""
BATCH_DIR=""
HOTEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline) PIPELINE="$2"; shift 2 ;;
    --config)   CONFIG="$2";   shift 2 ;;
    --batch)    BATCH_DIR="$2"; shift 2 ;;
    --hotel)    HOTEL="$2";    shift 2 ;;
    -h|--help)
      cat <<'HELP'
Usage:
  batch_run.sh --pipeline <petct_coreg_atlas.sh> \
               --config   <config.yaml>           \
               --batch    <batch_dir>              \
               --hotel    {1|3|4}

Arguments:
  --pipeline   Path to petct_coreg_atlas.sh
  --config     Path to config.yaml (shared across all runs)
  --batch      Folder containing one subdirectory per acquisition
  --hotel      Bed configuration: 1 (single), 3 (hotel x3), 4 (hotel x4)

Each subdirectory in <batch_dir> must contain:
  - *PET*.dcm and *CT*.dcm
  - SUV.txt (hotel=1) or SUV_1.txt...SUV_N.txt (hotel>1)

Outputs:
  - <subdir>/output/             per-run results
  - <subdir>/output/pipeline.log per-run log
  - <batch_dir>/batch_QC_summary.csv consolidated QC table
HELP
      exit 0 ;;
    *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
  esac
done

# ---- Validate arguments ----
[[ -z "$PIPELINE"  ]] && { echo "ERROR: Missing --pipeline"; exit 1; }
[[ -z "$CONFIG"    ]] && { echo "ERROR: Missing --config";   exit 1; }
[[ -z "$BATCH_DIR" ]] && { echo "ERROR: Missing --batch";    exit 1; }
[[ -z "$HOTEL"     ]] && { echo "ERROR: Missing --hotel";    exit 1; }

[[ -f "$PIPELINE" ]] || { echo "ERROR: Pipeline script not found: $PIPELINE"; exit 1; }
[[ -f "$CONFIG"   ]] || { echo "ERROR: Config file not found: $CONFIG";       exit 1; }
[[ -d "$BATCH_DIR" ]] || { echo "ERROR: Batch directory not found: $BATCH_DIR"; exit 1; }

HOTEL="${HOTEL//[^0-9]/}"
if [[ "$HOTEL" -ne 1 && "$HOTEL" -ne 3 && "$HOTEL" -ne 4 ]]; then
  echo "ERROR: --hotel must be 1, 3 or 4"; exit 1
fi

PIPELINE="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$PIPELINE")"
CONFIG="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$CONFIG")"
BATCH_DIR="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$BATCH_DIR")"

log() { echo "[$(date '+%F %T')] $*"; }

# ---- Collect subdirectories ----
mapfile -t SUBDIRS < <(find "$BATCH_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
  echo "ERROR: No subdirectories found in ${BATCH_DIR}"; exit 1
fi

BATCH_QC="${BATCH_DIR}/batch_QC_summary.csv"
TOTAL=${#SUBDIRS[@]}
OK_RUNS=()
FAILED_RUNS=()
SKIPPED_RUNS=()

log "======================================================"
log "BATCH RUN"
log "  pipeline  : ${PIPELINE}"
log "  config    : ${CONFIG}"
log "  batch_dir : ${BATCH_DIR}"
log "  hotel     : ${HOTEL}"
log "  runs      : ${TOTAL}"
log "======================================================"

# Write consolidated QC header (overwrites any previous file)
echo "run_name,run_dir,position,suv_file,pet_ct_METRIC,brainmask_dice,ct_atlas_METRIC" > "$BATCH_QC"

IDX=0
for SUBDIR in "${SUBDIRS[@]}"; do
  IDX=$((IDX + 1))
  RUN_NAME="$(basename "$SUBDIR")"

  log "------------------------------------------------------"
  log "Run ${IDX}/${TOTAL}: ${RUN_NAME}"
  log "------------------------------------------------------"

  # Check DICOMs are present
  shopt -s nullglob
  PET_CHECK=("${SUBDIR}"/*PET*.dcm)
  CT_CHECK=("${SUBDIR}"/*CT*.dcm)
  shopt -u nullglob

  if [[ ${#PET_CHECK[@]} -eq 0 || ${#CT_CHECK[@]} -eq 0 ]]; then
    log "WARN: No DICOMs in ${RUN_NAME} -> skipping"
    SKIPPED_RUNS+=("${RUN_NAME} (no DICOMs)")
    continue
  fi

  # Per-run output directory and log
  RUN_OUT="${SUBDIR}/output"
  RUN_LOG="${RUN_OUT}/pipeline.log"
  mkdir -p "$RUN_OUT"

  log "  subdir : ${SUBDIR}"
  log "  output : ${RUN_OUT}"
  log "  log    : ${RUN_LOG}"

  # Run the pipeline; capture exit code without stopping the batch
  RUN_STATUS=0
  (
    cd "$SUBDIR"
    bash "$PIPELINE" --config "$CONFIG" --hotel "$HOTEL"
  ) 2>&1 | tee "$RUN_LOG" || RUN_STATUS=${PIPESTATUS[0]}

  if [[ $RUN_STATUS -ne 0 ]]; then
    log "ERROR: FAILED for ${RUN_NAME} (exit ${RUN_STATUS})"
    FAILED_RUNS+=("$RUN_NAME")
  else
    log "OK: completed for ${RUN_NAME}"
    OK_RUNS+=("$RUN_NAME")
  fi

  # Append this run's QC rows to the consolidated table
  RUN_QC="${RUN_OUT}/QC_summary.csv"
  if [[ -f "$RUN_QC" ]]; then
    tail -n +2 "$RUN_QC" | while IFS= read -r line; do
      echo "${RUN_NAME},${line}"
    done >> "$BATCH_QC"
  fi

done  # end loop

# ---- Summary ----
log "======================================================"
log "BATCH COMPLETE"
log "  Total   : ${TOTAL}"
log "  OK      : ${#OK_RUNS[@]}"
log "  Failed  : ${#FAILED_RUNS[@]}"
log "  Skipped : ${#SKIPPED_RUNS[@]}"
[[ ${#FAILED_RUNS[@]}  -gt 0 ]] && log "  Failed runs  : ${FAILED_RUNS[*]}"
[[ ${#SKIPPED_RUNS[@]} -gt 0 ]] && log "  Skipped runs : ${SKIPPED_RUNS[*]}"
log "  Batch QC: ${BATCH_QC}"
log "======================================================"

[[ ${#FAILED_RUNS[@]} -eq 0 ]] || exit 1