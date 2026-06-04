#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Automated CT-based pipeline for mouse brain PET-CT quantification
#
# Performs atlas-normalized regional SUV extraction from raw DICOM data in a
# fully automated, reproducible workflow. Supports single-animal and
# multi-animal ("mouse-hotel") acquisition configurations.
#
# Processing steps:
#   1) DICOM -> NIfTI (PET and CT separately) via dcm2niix
#   2) Field-of-view split for multi-animal beds (hotel=1/3/4) via c3d -region
#   3) Per animal position:
#        - SUV scaling (c3d -scale)
#        - PET -> CT rigid registration (ANTs, MI)
#        - CT brain mask via nnU-Net (HU clip), mask dilation, apply to CT and PET
#        - Trim 1 voxel of background after masking
#        - CT -> Atlas registration (ANTs: Rigid MI + Affine MI + SyN CC r=6)
#        - Apply CT->Atlas transforms to PET SUV image
#        - Regional SUV statistics per atlas label (c3d -lstat)
#
# Output structure:
#   - <output.dir>/cropped/           split PET/CT per position
#   - <output.dir>/HOTEL_POS_i/       full per-position processing results
#   - <run_dir>/logs/                 DICOM conversion logs
#   - QC outputs (optional):
#       * PNG overlays: PET->CT alignment, CT->Atlas registration with labels
#       * QC_summary.csv: MI PET->CT, Dice brain mask, MI CT->Atlas
#
# Usage:
#   ./petct_coreg_atlas.sh --config config.yaml --hotel {1|3|4}
#
# See README.md and config_template.yaml for setup instructions.
# ==============================================================================

CONFIG=""
HOTEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --hotel)  HOTEL="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage:
  petct_coreg_atlas.sh --config config.yaml --hotel {1|3|4}

Run directory must contain:
  - DICOMs: *PET*.dcm and *CT*.dcm
  - SUV files:
      * hotel=1: SUV.txt
      * hotel>1: SUV_1.txt ... SUV_4.txt (as many as positions)
    Backward-compatible fallback:
      * any SUV* files (lexicographically sorted) will be assigned to POS_1..POS_N

Output structure:
  - <output.dir>/cropped/              (split PET/CT: 01.*_POS_i.nii.gz)
  - <output.dir>/HOTEL_POS_i/          (pipeline per position)
  - <output.dir>/QC_summary.csv        (if qc.enabled && qc.write_csv)
  - <run_dir>/logs/                    (global conversion logs)
EOF
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -z "${CONFIG}" ]] && { echo "ERROR: Missing --config"; exit 1; }
[[ -z "${HOTEL}"  ]] && { echo "ERROR: Missing --hotel";  exit 1; }

log(){ echo "[$(date '+%F %T')] $*"; }

is_true() {
  local v="${1:-}"
  v="${v,,}"
  [[ "$v" == "true" || "$v" == "1" || "$v" == "yes" ]]
}

abs_path() {
  python3 - "$1" <<'PY'
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
}

command -v python3 >/dev/null || { echo "ERROR: python3 required"; exit 1; }

# --- sanitize args ---
CONFIG="$(printf '%s' "$CONFIG" | tr -d '\r' | xargs)"
HOTEL_RAW="$(printf '%s' "$HOTEL" | tr -d '\r' | xargs)"
HOTEL="${HOTEL_RAW//[^0-9]/}"
[[ -n "$HOTEL" ]] || { echo "ERROR: invalid --hotel (received: '$HOTEL_RAW')"; exit 1; }
HOTEL=$((10#$HOTEL))
if [[ "$HOTEL" -ne 1 && "$HOTEL" -ne 3 && "$HOTEL" -ne 4 ]]; then
  echo "ERROR: --hotel must be 1, 3 or 4"; exit 1
fi

# ---------------- YAML getters ----------------
pyget() {
  python3 - "$CONFIG" "$1" <<'PY'
import yaml,sys
cfg=yaml.safe_load(open(sys.argv[1]))
v=cfg
for k in sys.argv[2].split('.'):
    v=v[k]
print(v)
PY
}

pyget_default() {
  python3 - "$CONFIG" "$1" "$2" <<'PY'
import yaml,sys
cfg=yaml.safe_load(open(sys.argv[1]))
v=cfg
for k in sys.argv[2].split('.'):
    if not isinstance(v, dict) or k not in v:
        print(sys.argv[3]); sys.exit(0)
    v=v[k]
print(v)
PY
}

# ---------------- Run dirs ----------------
RUN_DIR="$(pwd)"
GLOBAL_LOG_DIR="$(abs_path "${RUN_DIR}/logs")"
OUT_DIR="$(pyget_default output.dir "$(abs_path "${RUN_DIR}/output")")"
OUT_DIR="$(abs_path "$OUT_DIR")"
mkdir -p "$GLOBAL_LOG_DIR" "$OUT_DIR"

CROPPED_DIR="${OUT_DIR}/cropped"
mkdir -p "$CROPPED_DIR"

# ---------------- Tools ----------------
DCM2NIIX="$(pyget_default tools.dcm2niix "dcm2niix")"
C3D="$(pyget_default tools.c3d "c3d")"
ANTSREG="$(pyget_default tools.antsRegistration "antsRegistration")"
ANTSAPPLY="$(pyget_default tools.antsApplyTransforms "antsApplyTransforms")"
IMG_MATH="$(pyget_default tools.ImageMath "ImageMath")"
FSLSTATS="$(pyget_default tools.fslstats "")"
NNPRED="$(pyget_default tools.nnUNetv2_predict "nnUNetv2_predict")"
MEASURE_MI="$(pyget_default tools.MeasureImageSimilarity "")"

# ---------------- Atlas ----------------
ATLAS_HOME="$(pyget atlas.home)"
ATLAS_CT="$(pyget atlas.ct)"
ATLAS_LB="$(pyget atlas.labels)"
ATLAS_BM="$(pyget_default atlas.brain_mask "")"

# ---------------- nnUNet ----------------
NN_RESULTS_DIR="$(pyget_default nnunet.results_dir "")"
NN_DATASET="$(pyget_default nnunet.dataset "")"
NN_CONF="$(pyget_default nnunet.configuration "2d")"
NN_TRAINER="$(pyget_default nnunet.trainer "nnUNetTrainer")"
NN_PLANS="$(pyget_default nnunet.plans "nnUNetPlans")"
NN_DEVICE="$(pyget_default nnunet.device "cpu")"
NN_NPROC="$(pyget_default nnunet.num_processes "1")"
NN_CONTINUE="$(pyget_default nnunet.continue_prediction "true")"
NN_FOLDS_CFG="$(pyget_default nnunet.folds "0 1 2 3 4")"
NN_SCRATCH_DIR="$(pyget_default nnunet.scratch_dir "")"

# Optional conda activation (only if provided)
CONDA_ACTIVATE_SCRIPT="$(pyget_default conda.activate_script "")"
CONDA_ENV="$(pyget_default conda.env "")"
activate_env_if_requested() {
  if [[ -n "$CONDA_ACTIVATE_SCRIPT" && -n "$CONDA_ENV" ]]; then
    # shellcheck disable=SC1090
    . "$CONDA_ACTIVATE_SCRIPT"
    conda activate "$CONDA_ENV"
  fi
}
deactivate_env_if_requested() {
  if [[ -n "$CONDA_ACTIVATE_SCRIPT" && -n "$CONDA_ENV" ]]; then
    conda deactivate || true
  fi
}

# runtime threads (optional)
ITK_THREADS="$(pyget_default runtime.itk_threads "")"
if [[ -n "$ITK_THREADS" ]]; then
  export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$ITK_THREADS"
  export OMP_NUM_THREADS="$ITK_THREADS"
fi

# ---------------- QC ----------------
QC_ENABLED="$(pyget_default qc.enabled "true")"
QC_WRITE_CSV="$(pyget_default qc.write_csv "true")"
QC_WRITE_IMAGES="$(pyget_default qc.write_images "true")"
QC_IMG_DIRNAME="$(pyget_default qc.images_dirname "qc_images")"
QC_SUMMARY="${OUT_DIR}/QC_summary.csv"

# ---------------- Run summary (informational) ----------------
log "RUN_DIR      = ${RUN_DIR}"
log "OUT_DIR      = ${OUT_DIR}"
log "CROPPED_DIR  = ${CROPPED_DIR}"
log "GLOBAL_LOGS  = ${GLOBAL_LOG_DIR}"
log "HOTEL        = ${HOTEL}"
log "ATLAS_CT     = ${ATLAS_HOME}/${ATLAS_CT}"
log "ATLAS_LABELS = ${ATLAS_HOME}/${ATLAS_LB}"
log "ATLAS_BM     = ${ATLAS_HOME}/${ATLAS_BM:-NONE}"
log "QC_ENABLED   = ${QC_ENABLED} (csv=${QC_WRITE_CSV}, images=${QC_WRITE_IMAGES})"
log "ITK_THREADS  = ${ITK_THREADS:-default}"

# =========================================================
# QC image renderer (Python)
# =========================================================
render_qc_png() {
  local fixed="$1"
  local moving="$2"
  local labels="$3"
  local outpng="$4"
  local title="$5"

  python3 - "$fixed" "$moving" "$labels" "$outpng" "$title" <<'PY' 2>/dev/null || true
import sys,os,numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fixed,moving,labels,outpng,title=sys.argv[1:]
def load(p):
    if p=="EMPTY" or (not p) or (not os.path.exists(p)):
        return None
    x = np.nan_to_num(nib.load(p).get_fdata())
    # Some atlas resources are stored as 4D NIfTI with a singleton last dimension (e.g., (X,Y,Z,1)).
    # Squeeze to 3D to keep matplotlib happy.
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    return x
F=load(fixed); M=load(moving); L=load(labels)
if F is None: raise RuntimeError("fixed image not found")
cx,cy,cz=[s//2 for s in F.shape[:3]]
def norm(x):
    nz=x[x!=0]
    if nz.size: lo,hi=np.percentile(nz,[1,99])
    else: lo,hi=0,1
    y=(x-lo)/(hi-lo+1e-6)
    return np.clip(y,0,1)
slices=[F[cx,:,:].T,F[:,cy,:].T,F[:,:,cz].T]
fig,ax=plt.subplots(1,3,figsize=(12,4),dpi=150)
for i,s in enumerate(slices):
    ax[i].imshow(norm(s),cmap="gray",origin="lower")
    if M is not None:
        ms=[M[cx,:,:].T,M[:,cy,:].T,M[:,:,cz].T][i]
        ax[i].imshow(norm(ms),cmap="hot",alpha=0.45,origin="lower")
    if L is not None:
        ls=[L[cx,:,:].T,L[:,cy,:].T,L[:,:,cz].T][i]
        ax[i].contour(ls>0,[0.5],colors="cyan",linewidths=0.8)
    ax[i].axis("off")
fig.suptitle(title)
os.makedirs(os.path.dirname(outpng),exist_ok=True)
fig.savefig(outpng)
PY
}

# =========================================================
# Helpers: Dice + MI/CC (optional)
# =========================================================
dice_safe() {
  local m1="$1"
  local m2="$2"
  local d=""
  d=$("$C3D" "$m1" "$m2" -overlap 1 2>/dev/null | awk -F'[, ]+' '/^OVL:/{print $(NF-1); exit}' || true)
  [[ -z "$d" ]] && d="NA"
  echo "$d"
}
mis_safe_mask() {
  local metric="$1"
  local mask="$2"
  local errlog="$3"
  if [[ -z "$MEASURE_MI" || ! -x "$MEASURE_MI" ]]; then
    echo "NA"; return 0
  fi
  local out=""
  out=$("$MEASURE_MI" -d 3 -m "$metric" -x "[${mask},${mask}]" -v 0 2>>"$errlog"         | tail -n 1 | tr -d '[:space:]' || true)
  [[ -z "$out" ]] && out="NA"
  echo "$out"
}

# =========================================================
# STEP 0: Validate inputs
# =========================================================
log "STEP 0: Validating inputs in ${RUN_DIR}"
log "Looking for DICOM patterns: *PET*.dcm and *CT*.dcm"
log "Looking for SUV files: SUV.txt / SUV_1.txt..SUV_N.txt (fallback: SUV*)"

shopt -s nullglob # Prevents literal pattern if no match
PET_DCMS=(*PET*.dcm)
CT_DCMS=(*CT*.dcm)

shopt -u nullglob
[[ ${#PET_DCMS[@]} -ge 1 ]] || { echo "ERROR: No *PET*.dcm found"; exit 1; }
[[ ${#CT_DCMS[@]}  -ge 1 ]] || { echo "ERROR: No *CT*.dcm found"; exit 1; }

# =========================================================
# STEP 1: CONVERT DICOM -> NIfTI (dcm2niix)
#   Converts PET and CT DICOM series independently to NIfTI format.
#   PET and CT are processed in separate temporary directories to prevent
#   cross-contamination when both modalities share the same run directory.
#   Reference: https://github.com/rordenlab/dcm2niix
# =========================================================
log "STEP 1: DICOM -> NIfTI (dcm2niix)"
log "Outputs: 01.PET.nii.gz and 01.CT.nii.gz"

rm -f 01.PET.nii* 01.CT.nii* *.json 2>/dev/null || true
rm -rf __tmp_pet __tmp_ct 2>/dev/null || true
mkdir -p __tmp_pet __tmp_ct
for f in "${PET_DCMS[@]}"; do cp -f "$f" __tmp_pet/; done
for f in "${CT_DCMS[@]}";  do cp -f "$f" __tmp_ct/;  done
(
  cd __tmp_pet
  "$DCM2NIIX" -z y -s y -f PET . 2>&1 | tee "${GLOBAL_LOG_DIR}/dcm2niix_pet.log"
)
(
  cd __tmp_ct
  "$DCM2NIIX" -z y -s y -f CT . 2>&1 | tee "${GLOBAL_LOG_DIR}/dcm2niix_ct.log"
)

# --- OPTIONAL WARNING: multiple NIfTI candidates generated ---
shopt -s nullglob
PET_CANDS=(__tmp_pet/PET*.nii.gz)
CT_CANDS=(__tmp_ct/CT*.nii.gz)
shopt -u nullglob

if [[ ${#PET_CANDS[@]} -gt 1 ]]; then
  log "WARN: Multiple PET NIfTIs; using first: ${PET_CANDS[0]}"
fi
if [[ ${#CT_CANDS[@]} -gt 1 ]]; then
  log "WARN: Multiple CT NIfTIs; using first: ${CT_CANDS[0]}"
fi

PET_NII="$(ls -1 __tmp_pet/PET*.nii.gz 2>/dev/null | head -n1 || true)"
CT_NII="$(ls -1 __tmp_ct/CT*.nii.gz  2>/dev/null | head -n1 || true)"

[[ -n "$PET_NII" ]] || { echo "ERROR: PET*.nii.gz not generated"; exit 1; }
[[ -n "$CT_NII"  ]] || { echo "ERROR: CT*.nii.gz not generated"; exit 1; }
mv -f "$PET_NII" 01.PET.nii.gz
mv -f "$CT_NII"  01.CT.nii.gz
rm -rf __tmp_pet __tmp_ct
rm -f *.json 2>/dev/null || true
log "OK: 01.PET.nii.gz and 01.CT.nii.gz created"

# =========================================================
# STEP 2: Split HOTEL -> <out>/cropped
# =========================================================
log "STEP 2: HOTEL split = ${HOTEL}"
log "Split is done in XY using fixed percentage regions based on bed geometry"
log "Outputs -> ${CROPPED_DIR}/01.{CT,PET}_POS_i.nii.gz"

rm -f "${CROPPED_DIR}/01.CT_POS_"*.nii.gz "${CROPPED_DIR}/01.PET_POS_"*.nii.gz 2>/dev/null || true
if [[ "$HOTEL" -eq 4 ]]; then
  log "HOTEL size -> 4"
  # split CT
  "$C3D" 01.CT.nii.gz  -region 50%x50%x0% 50%x50%x100% -o "${CROPPED_DIR}/01.CT_POS_1.nii.gz"
  "$C3D" 01.CT.nii.gz  -region 0%x50%x0%  50%x50%x100% -o "${CROPPED_DIR}/01.CT_POS_2.nii.gz"
  "$C3D" 01.CT.nii.gz  -region 0%x0%x0%   50%x50%x100% -o "${CROPPED_DIR}/01.CT_POS_3.nii.gz"
  "$C3D" 01.CT.nii.gz  -region 50%x0%x0%  50%x50%x100% -o "${CROPPED_DIR}/01.CT_POS_4.nii.gz"

  # split PET
  "$C3D" 01.PET.nii.gz -region 50%x50%x0% 50%x50%x100% -o "${CROPPED_DIR}/01.PET_POS_1.nii.gz"
  "$C3D" 01.PET.nii.gz -region 0%x50%x0%  50%x50%x100% -o "${CROPPED_DIR}/01.PET_POS_2.nii.gz"
  "$C3D" 01.PET.nii.gz -region 0%x0%x0%   50%x50%x100% -o "${CROPPED_DIR}/01.PET_POS_3.nii.gz"
  "$C3D" 01.PET.nii.gz -region 50%x0%x0%  50%x50%x100% -o "${CROPPED_DIR}/01.PET_POS_4.nii.gz"

elif [[ "$HOTEL" -eq 3 ]]; then
  log "HOTEL size -> 3"
  # split CT
  "$C3D" 01.CT.nii.gz  -region 50%x50%x0% 50%x50%x100% -o "${CROPPED_DIR}/01.CT_POS_1.nii.gz"
  "$C3D" 01.CT.nii.gz  -region 0%x50%x0%  50%x50%x100% -o "${CROPPED_DIR}/01.CT_POS_2.nii.gz"
  "$C3D" 01.CT.nii.gz  -region 25%x0%x0%  50%x50%x100% -o "${CROPPED_DIR}/01.CT_POS_3.nii.gz"

  # split PET
  "$C3D" 01.PET.nii.gz -region 50%x50%x0% 50%x50%x100% -o "${CROPPED_DIR}/01.PET_POS_1.nii.gz"
  "$C3D" 01.PET.nii.gz -region 0%x50%x0%  50%x50%x100% -o "${CROPPED_DIR}/01.PET_POS_2.nii.gz"
  "$C3D" 01.PET.nii.gz -region 25%x0%x0%  50%x50%x100% -o "${CROPPED_DIR}/01.PET_POS_3.nii.gz"

elif [[ "$HOTEL" -eq 1 ]]; then
  log "HOTEL size -> 1"
  # Single animal: no splitting needed; copy the full volume as POS_1
  cp -f 01.CT.nii.gz  "${CROPPED_DIR}/01.CT_POS_1.nii.gz"
  cp -f 01.PET.nii.gz "${CROPPED_DIR}/01.PET_POS_1.nii.gz"

else
  echo "ERROR: internal: HOTEL must be 1, 3, or 4"; exit 1
fi

# ====================================================================================
# STEP 3: PREP — folders + copies + SUV mapping
#   - Create per-position output folders: <OUT_DIR>/HOTEL_POS_i/
#   - Copy split CT/PET into each position folder as 01.CT.nii.gz / 01.PET.nii.gz
#   - Map SUV scaling factors to each position (POS_i) using:
#       1) Preferred: SUV.txt (HOTEL=1) or SUV_1.txt..SUV_N.txt (HOTEL>1)
#       2) Fallback (legacy): any SUV* files (sorted) assigned to POS_1..POS_N
#   - If a position has no SUV file, we keep the folder but skip quantitative steps
#     (no SUV scaling + no SUV_values.csv for that position).
# ====================================================================================
log "STEP 3: Preparing HOTEL_POS_i folders"

# ---------------- SUV file mapping per position ----------------
declare -a SUV_BY_POS
for ((i=1; i<=HOTEL; i++)); do SUV_BY_POS[$i]=""; done

# Preferred naming conventions (explicit mapping)
if [[ "$HOTEL" -eq 1 ]]; then
  [[ -f "${RUN_DIR}/SUV.txt" ]] && SUV_BY_POS[1]="${RUN_DIR}/SUV.txt"
fi
if [[ "$HOTEL" -gt 1 ]]; then
  for ((i=1; i<=HOTEL; i++)); do
    [[ -f "${RUN_DIR}/SUV_${i}.txt" ]] && SUV_BY_POS[$i]="${RUN_DIR}/SUV_${i}.txt"
  done
fi

# Fallback to legacy SUV* ordering if mapping is missing
if [[ -z "${SUV_BY_POS[1]}" ]]; then
  shopt -s nullglob
  SUV_LIST=(SUV*)
  shopt -u nullglob
  IFS=$'\n' SUV_LIST=($(printf "%s\n" "${SUV_LIST[@]}" | sort)); unset IFS
  
  for ((i=1; i<=HOTEL; i++)); do
    idx=$((i-1))
    [[ $idx -lt ${#SUV_LIST[@]} ]] && SUV_BY_POS[$i]="${RUN_DIR}/${SUV_LIST[$idx]}"
  done
fi

# Print SUV mapping (useful for debugging / reproducibility)
for ((i=1; i<=HOTEL; i++)); do
  log "SUV map POS_${i}: ${SUV_BY_POS[$i]:-NONE}"
done

# ---------------- QC summary CSV header (written once) ----------------
if is_true "$QC_ENABLED" && is_true "$QC_WRITE_CSV"; then
  if [[ ! -f "$QC_SUMMARY" ]]; then
    echo "run_dir,position,suv_file,pet_ct_METRIC,brainmask_dice,ct_atlas_METRIC" > "$QC_SUMMARY"
  fi
fi

# ---------------- Create per-position folders + stage inputs ----------------
for ((i=1; i<=HOTEL; i++)); do
  POS_DIR="${OUT_DIR}/HOTEL_POS_${i}"
  mkdir -p "$POS_DIR" "${POS_DIR}/logs"

  CT_SRC="${CROPPED_DIR}/01.CT_POS_${i}.nii.gz"
  PET_SRC="${CROPPED_DIR}/01.PET_POS_${i}.nii.gz"

  # If split images are missing, mark and skip this position
  if [[ ! -f "$CT_SRC" || ! -f "$PET_SRC" ]]; then
    echo "WARN: Missing split images for POS_${i}. Skipping."
    echo "MISSING_CROPPED_IMAGES" > "${POS_DIR}/NO_CROPPED_SKIP_PIPELINE.txt"
    continue
  fi

  cp -f "$CT_SRC"  "${POS_DIR}/01.CT.nii.gz"
  cp -f "$PET_SRC" "${POS_DIR}/01.PET.nii.gz"

 # If SUV file is missing for a position, create a marker file to skip quantification
  SUV_SRC="${SUV_BY_POS[$i]}"
  if [[ -n "$SUV_SRC" && -f "$SUV_SRC" ]]; then
    cp -f "$SUV_SRC" "${POS_DIR}/SUV.txt"
  else
    log "INFO: No SUV file for POS_${i}. Skipping quantitative processing."
    echo "NO_SUV_FOR_THIS_POSITION" > "${POS_DIR}/NO_SUV_SKIP_PIPELINE.txt"
  fi
done

# =========================================================
# STEP 4: Full pipeline per position
#   Loop over each HOTEL bed position (POS_1..POS_N) and run:
#     - SUV scaling (PET -> PETSUV)
#     - PET->CT registration
#     - Brain masking (nnUNet) + trim
#     - CT->Atlas registration + transform PET to atlas space
#     - ROI stats + optional QC
#
#   Positions can be skipped if:
#     - split PET/CT was not generated (NO_CROPPED_SKIP_PIPELINE.txt)
#     - no SUV factor provided for that position (NO_SUV_SKIP_PIPELINE.txt)
# =========================================================
log "STEP 4: Full pipeline per position"
for ((i=1; i<=HOTEL; i++)); do
  POS_DIR="${OUT_DIR}/HOTEL_POS_${i}"
  [[ -d "$POS_DIR" ]] || continue

  # Skip rules (leave a marker file in STEP 3)
  [[ -f "${POS_DIR}/NO_SUV_SKIP_PIPELINE.txt" ]] && { log "POS_${i}: missing SUV -> skip"; continue; }
  [[ -f "${POS_DIR}/NO_CROPPED_SKIP_PIPELINE.txt" ]] && { log "POS_${i}: missing split -> skip"; continue; }

  log "========================================================="
  log "Processing HOTEL_POS_${i} / ${HOTEL}"
  log "========================================================="
  cd "$POS_DIR"

  # ---------------------------------------------------------
  # Read SUV scaling factor from SUV.txt
  #   - accepts decimal comma (e.g., '1,23' -> '1.23')
  #   - must be a positive float
  # This value is used as a multiplicative factor:
  #   PETSUV = PET * SUV_VAL
  # ---------------------------------------------------------
  SUV_VAL="$(head -n1 SUV.txt | tr -d '\r')"
  SUV_VAL="${SUV_VAL//,/.}"

  # Validate SUV as a positive numeric value (fail early if invalid)
  if ! python3 - <<PY
import sys
try:
    v=float("${SUV_VAL}")
    assert v>0
except Exception:
    sys.exit(1)
PY
  then
    echo "ERROR: Invalid SUV value '${SUV_VAL}' in ${POS_DIR}/SUV.txt"
    exit 1
  fi
  log "SUV detected: ${SUV_VAL}"

  # ---------------------------------------------------------
  # PET SUV scaling
  # Output: 01.PETSUV.nii.gz  (PET intensity scaled by SUV factor)
  # ---------------------------------------------------------
  log "Read SUV.txt and scale PET -> 01.PETSUV.nii.gz"
  "$C3D" 01.PET.nii.gz -scale "$SUV_VAL" -o 01.PETSUV.nii.gz


# ======================================================================================
# STEP 4A: RIGID PET -> CT COREGISTRATION
#   Goal:
#     Align PET (SUV-scaled) to the CT of the same position using rigid registration.
#
#   Why rigid?
#     PET and CT are acquired on the same bed; main differences are translations/rotations between modalities.
#
#   Fixed / Moving definition:
#     - Fixed  image: CT  (01.CT.nii.gz)
#     - Moving image: PETSUV (01.PETSUV.nii.gz)
#
#     The output 01.SUV2CT.nii.gz is the moving image (PETSUV) resampled into the fixed image space (CT).
#
#   Registration details (ANTs):
#     - Transform: Rigid (6 DOF)
#     - Similarity metric: MI (Mutual Information), robust for multi-modality (PET vs CT)
#     - No histogram matching (keep intensities as-is)
#     - Winsorization: clip extreme intensities to stabilize optimization
#
#   Outputs:
#     - PET2CT_0GenericAffine.mat  (rigid transform)
#     - 01.SUV2CT.nii.gz           (moving PETSUV resampled into CT space)
#     - logs/rigid_pet2ct.log      (full antsRegistration log)
# ======================================================================================
  log "STEP 4A: Rigid PET->CT (ANTs, MI) [core]"
  "$ANTSREG" \
    --verbose 1 \
    --dimensionality 3 \
    --collapse-output-transforms 1 \
    --output [ PET2CT_, 01.SUV2CT.nii.gz, 01.CT2PET.nii.gz ] \
    --interpolation Linear \
    --use-histogram-matching 0 \
    --winsorize-image-intensities [ 0.005,0.995 ] \
    --transform Rigid[ 0.1 ] \
    --metric MI[ 01.CT.nii.gz, 01.PETSUV.nii.gz, 1,32,Regular,0.25 ] \
    --convergence [ 1000x500x250x100,1e-6,10 ] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox \
    2>&1 | tee "${POS_DIR}/logs/rigid_pet2ct.log"


  # Inverse-resampled output is not required downstream; keep directory clean
  rm -f 01.CT2PET.nii.gz 2>/dev/null || true

# --------------------------------------------------------------------------------
# QC (optional): compute MI PET->CT post-registration using MeasureImageSimilarity
#   - Applied to the registered PET (01.SUV2CT.nii.gz) and CT (01.CT.nii.gz)
#   - Consistent with MI CT->Atlas metric (both use MeasureImageSimilarity)
# --------------------------------------------------------------------------------
  PET_CT_METRIC="NA"
  if is_true "$QC_ENABLED"; then
    if [[ -n "$MEASURE_MI" && -x "$MEASURE_MI" && -f "01.SUV2CT.nii.gz" && -f "01.CT.nii.gz" ]]; then
      PET_CT_METRIC="$("$MEASURE_MI" -d 3 -m "MI[01.CT.nii.gz,01.SUV2CT.nii.gz,1,32]" -v 0 2>"${POS_DIR}/logs/qc_pet_ct_mis_err.log" | tail -n 1 | tr -d '[:space:]' || true)"
    fi
    [[ -z "$PET_CT_METRIC" ]] && PET_CT_METRIC="NA"

  # Start per-position QC report file
    { echo "---- QC metrics ----"; echo "PET_CT_regMetricValue: ${PET_CT_METRIC}"; } > QC_metrics.txt
  fi

# ---------------------------------------------------------
# QC (optional): visual overlay (CT in gray + PET(SUV2CT) as hot overlay)
#   Output: qc_images/PET_CT_overlay.png
# ---------------------------------------------------------
  if is_true "$QC_ENABLED" && is_true "$QC_WRITE_IMAGES"; then
   mkdir -p "${QC_IMG_DIRNAME}"
   render_qc_png \
     "01.CT.nii.gz" \
     "01.SUV2CT.nii.gz" \
     "EMPTY" \
     "${QC_IMG_DIRNAME}/PET_CT_overlay.png" \
     "PET->CT (Rigid)"
  fi

# =============================================================================
# STEP 4B: GET CT BRAIN MASK (nnU-Net) + APPLY MASK + TRIM
#
# Goal:
#   Generate a brain mask from the CT using nnU-Net, then use it to:
#     1) isolate brain-only CT (for CT->Atlas registration)
#     2) isolate brain-only PET-in-CT (to later warp PET to atlas)
#
# Sub-steps:
#   4B.1  CT intensity clipping (HU range) to stabilize nnU-Net inference
#         - Input : 01.CT.nii.gz
#         - Output: 01.CT_scaled.nii.gz
#         - Note  : clip [-1000, 10000] HU (keeps most anatomy; reduces extremes)
#
#   4B.2  Prepare nnU-Net input folder + required filename convention
#         - nnU-Net expects "<case>_0000.nii.gz" for single-channel input
#         - We copy CT_scaled -> ${NN_WORKDIR}/ct_0000.nii.gz
#         - NN_WORKDIR can be a scratch location to avoid clutter + speed up I/O
#
#   4B.3  nnU-Net inference (brain vs non-brain segmentation)
#         - Output expected from nnU-Net: ${NN_WORKDIR}/ct.nii.gz
#         - We save it as: 02.brainMask_ia.nii.gz
#           ("ia" = initial/AI mask, before dilation)
#         - Folds can be explicit ("0 1 2 3 4") or auto-detected from model folder
#
#   4B.4  Optional sanity check: voxel count of the predicted mask (fslstats -V)
#         - Useful to catch empty/failed segmentations early
#
#   4B.5  Morphological dilation of the AI mask (ImageMath MD)
#         - Output: 02.brainMask.nii.gz
#         - Purpose: make mask more conservative (include edges / avoid cutting brain)
#         - Parameter "24" controls dilation radius/extent
#
#   4B.6  Apply brain mask to CT and PET-in-CT
#         - CT brain-only:     03.brainCT.nii.gz
#         - PET brain-only:    03.brainSUV2CT.nii.gz
#         - Operation: multiply image * mask (mask is 0/1)
#
#   4B.7  Trim 1 voxel AFTER masking
#         - Outputs:
#             03.brainCT_trim.nii.gz
#             03.brainSUV2CT_trim.nii.gz
#         - Note: trimming changes image dimensions slightly; do NOT move this earlier
#                 if you want identical quantitative outputs vs the original pipeline.
#
# Outputs of STEP 4B:
#   - 01.CT_scaled.nii.gz             (clipped CT for nnU-Net)
#   - 02.brainMask_ia.nii.gz          (raw nnU-Net mask)
#   - 02.brainMask.nii.gz             (dilated mask used for masking)
#   - 03.brainCT_trim.nii.gz          (brain-only CT, trimmed; used for CT->Atlas)
#   - 03.brainSUV2CT_trim.nii.gz      (brain-only PET in CT space, trimmed; later warped)
#   - logs/nnunet_predict.log         (nnU-Net inference log)
# =============================================================================
  log "STEP 4B: CT brain mask with nnU-Net"
  
  # Clip CT intensities to [-1000, 10000] HU for stable nnU-Net inference
  "$C3D" 01.CT.nii.gz -clip -1000 10000 -o 01.CT_scaled.nii.gz
  TS="$(date '+%Y%m%d_%H%M%S')"
  if [[ -n "$NN_SCRATCH_DIR" ]]; then NN_WORKDIR="${NN_SCRATCH_DIR}/petct_${TS}_pos${i}_$$"; else NN_WORKDIR="${POS_DIR}/temp_ai"; fi
  # Create a temporary working directory for nnU-Net input/output
  # (uses scratch_dir if configured, otherwise a subfolder in the position directory)
  rm -rf "$NN_WORKDIR"; mkdir -p "$NN_WORKDIR"

  # nnU-Net expects input files named as <case>_0000.nii.gz (single-channel convention)
  cp 01.CT_scaled.nii.gz "${NN_WORKDIR}/ct_0000.nii.gz"
  [[ -n "$NN_RESULTS_DIR" ]] && export nnUNet_results="$NN_RESULTS_DIR"

  AVAILABLE_FOLDS=""
  MODEL_DIR="${NN_RESULTS_DIR}/${NN_DATASET}/${NN_TRAINER}__${NN_PLANS}__${NN_CONF}"
  if [[ "${NN_FOLDS_CFG}" == "auto" && -n "$NN_RESULTS_DIR" && -d "$MODEL_DIR" ]]; then
    AVAILABLE_FOLDS="$(ls -1 "$MODEL_DIR" 2>/dev/null | sed -n 's/^fold_//p' | tr '\n' ' ' | sed 's/  */ /g' | sed 's/ *$//')"
  fi
  if [[ "${NN_FOLDS_CFG}" == "auto" ]]; then NN_FOLDS_FINAL="${AVAILABLE_FOLDS:-0 1 2 3 4}"; else NN_FOLDS_FINAL="${NN_FOLDS_CFG}"; fi
  read -r -a NN_FOLDS_ARR <<< "$NN_FOLDS_FINAL"

  activate_env_if_requested

  # Run nnU-Net inference to predict the brain mask from the CT volume
  log "nnU-Net brain mask prediction"
  "$NNPRED" \
    -d "$NN_DATASET" \
    -i "${NN_WORKDIR}" \
    -o "${NN_WORKDIR}" \
    -f "${NN_FOLDS_ARR[@]}" \
    -tr "$NN_TRAINER" \
    -c "$NN_CONF" \
    -p "$NN_PLANS" \
    -device "$NN_DEVICE" \
    -npp "$NN_NPROC" \
    -nps "$NN_NPROC" \
    $(is_true "$NN_CONTINUE" && echo "--continue_prediction") \
    2>&1 | tee "${POS_DIR}/logs/nnunet_predict.log"

  deactivate_env_if_requested

  # Rename and move nnU-Net result to position output directory
  [[ -f "${NN_WORKDIR}/ct.nii.gz" ]] || { echo "ERROR: nnUNet did not generate ct.nii.gz"; exit 1; }
  cp -f "${NN_WORKDIR}/ct.nii.gz" 02.brainMask_ia.nii.gz
  rm -rf "$NN_WORKDIR"

  # Check that there are enough voxels in the brain mask
  if [[ -n "$FSLSTATS" && -x "$FSLSTATS" ]]; then
    voxel_count=$("$FSLSTATS" 02.brainMask_ia.nii.gz -V | awk '{print $1}')
    log "ai_voxels = ${voxel_count}"
  fi

  # Morphologically dilate the brain mask (radius = 24 voxels = 4.8 mm at 0.2 mm CT voxel size)
  # This ensures the mask covers the skull and surrounding tissue, not only the brain parenchyma,
  # which improves robustness of the subsequent CT-to-atlas registration.
  "$IMG_MATH" 3 02.brainMask.nii.gz MD 02.brainMask_ia.nii.gz 24

  # Apply the dilated brain mask to CT and PET images (voxel-wise multiplication by binary mask)
  "$C3D" 01.CT.nii.gz     02.brainMask.nii.gz -multiply -o 03.brainCT.nii.gz
  "$C3D" 01.SUV2CT.nii.gz 02.brainMask.nii.gz -multiply -o 03.brainSUV2CT.nii.gz

  # Trim empty background voxels, keeping a 1-voxel margin, to reduce image size before atlas registration
  "$C3D" 03.brainCT.nii.gz     -trim 1vox -o 03.brainCT_trim.nii.gz
  "$C3D" 03.brainSUV2CT.nii.gz -trim 1vox -o 03.brainSUV2CT_trim.nii.gz

# ===========================================================
# STEP 4C: CT -> ATLAS COREGISTRATION (ANTs)
#   - Stage 1: Rigid  (MI)
#   - Stage 2: Affine (MI)
#   - Stage 3: SyN    (CC, radius=6)
# Output prefix: AT_  -> AT_0GenericAffine.mat + AT_1Warp.nii.gz
# ===========================================================
  FIXED="${ATLAS_HOME}/${ATLAS_CT}"
  MOVING="03.brainCT_trim.nii.gz"
  
  log "STEP 4C: CT -> Atlas (Rigid + Affine + SyN)"
  log "  Stage 1: Rigid (MI) | Stage 2: Affine (MI) | Stage 3: SyN (CC r=6)"

  "$ANTSREG" \
    --verbose 1 \
    --dimensionality 3 \
    --collapse-output-transforms 1 \
    --output [ AT_, out_CT2atlas.nii.gz, out_CT2atlas_inverse.nii.gz ] \
    --interpolation Linear \
    --use-histogram-matching 0 \
    --winsorize-image-intensities [ 0.005,0.995 ] \
    --initial-moving-transform [ "$FIXED","$MOVING",0 ] \
    \
    --transform Rigid[ 0.1 ] \
    --metric MI[ "$FIXED","$MOVING",1,32,Regular,0.25 ] \
    --convergence [ 1000x500x250x100,1e-6,10 ] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox \
    \
    --transform Affine[ 0.1 ] \
    --metric MI[ "$FIXED","$MOVING",1,32,Regular,0.25 ] \
    --convergence [ 1000x500x250x100,1e-6,10 ] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox \
    \
    --transform SyN[ 0.1,3,0 ] \
    --metric CC[ "$FIXED","$MOVING",1,6 ] \
    --convergence [ 100x70x50x20,1e-6,10 ] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox \
    \
    2>&1 | tee "${POS_DIR}/logs/ct2atlas.log"

# ===========================================================
# STEP 4D: APPLY CT -> ATLAS TRANSFORMS TO SUV2CT IMAGE
#
# Goal:
#   Warp the PET image (already aligned to CT) into atlas space using the transforms computed in STEP 4C.
#
# Inputs:
#   - Moving image: 03.brainSUV2CT_trim.nii.gz  (PET in CT space trimmed)
#   - Reference:    atlas CT (defines atlas space geometry)
#
# Transforms applied (ANTs order):
#   1) AT_1Warp.nii.gz           (nonlinear SyN deformation)
#   2) AT_0GenericAffine.mat     (affine transform)
#
# Output:
#   - out_SUV2Atlas.nii.gz       (PET in atlas space)
#
# Interpolation:
#   Linear interpolation is used because PET contains continuous values.
# ===========================================================
  log "STEP 4D: Applying CT->Atlas transforms to SUV"
  "$ANTSAPPLY" \
    -d 3 \
    -r "${ATLAS_HOME}/${ATLAS_CT}" \
    -i 03.brainSUV2CT_trim.nii.gz \
    -n Linear \
    -t AT_1Warp.nii.gz \
    -t AT_0GenericAffine.mat \
    -o out_SUV2Atlas.nii.gz \
    2>&1 | tee "${POS_DIR}/logs/apply_suv.log"

  # Copy atlas labels locally for ROI statistics in STEP 4E
  cp -f "${ATLAS_HOME}/${ATLAS_LB}" out_labels.nii.gz

# ===========================================================
# STEP 4E: ROI STATISTICS (c3d -lstat) -> SUV_values.csv
#
# Goal:
#   Extract regional PET SUV values using atlas labels (Region Of Interest = ROI).
#
# Inputs:
#   - out_SUV2Atlas.nii.gz   (PET SUV image warped to atlas space)
#   - out_labels.nii.gz      (atlas label map; each integer = ROI)
#
# Method:
#   c3d -lstat computes statistics for each label region.
#
# Output file: SUV_values.csv
#   Columns written:
#     1) Label ID
#     2) Mean SUV
#     3) Number of voxels in the ROI
# ===========================================================
  log "STEP 4E: ROI stats in atlas space (labels -> SUV_values.csv)"
  log "Outputs: out_SUV2Atlas.nii.gz, SUV_values.csv"  

  # Extract PET SUV statistics per atlas label
  "$C3D" out_SUV2Atlas.nii.gz out_labels.nii.gz -lstat | awk '{print $1, $2, $3}' > SUV_values.csv

# ======================================================================
# STEP 5 (optional): QC: initialize per-position metrics
# These metrics provide optional quality control for the registration:
#
#   BMDICE            Dice overlap between AI brain mask (mapped to atlas space)
#                     and atlas brain mask (if provided).
#
#   CT_ATLAS_METRIC   Mutual Information between CT and atlas CT
#                     after full registration (Rigid + Affine + SyN).
#
# Default value:
#   NA = metric not computed (e.g., QC disabled or required tools missing).
# ======================================================================
  if is_true "$QC_ENABLED"; then
    log "STEP 5: QC metrics"
      BMDICE="NA"
      PET_CT_METRIC="${PET_CT_METRIC:-NA}"
      CT_ATLAS_METRIC="NA"
  else
    log "STEP 5: QC metrics skipped (qc.enabled=false)"
  fi

# =========================================================
# QC STEP 5A: CT->Atlas visual overlay
#   - fixed: atlas CT (gray)
#   - moving: warped CT (hot overlay)
#   - labels: atlas labels (cyan contour)
# Output: qc_images/CT_Atlas_labels.png
# =========================================================
  if is_true "$QC_ENABLED" && is_true "$QC_WRITE_IMAGES"; then
    log "QC 5A: Writing CT->Atlas + labels overlay PNG"
    mkdir -p "${QC_IMG_DIRNAME}"
    render_qc_png \
      "${ATLAS_HOME}/${ATLAS_CT}" \
      "out_CT2atlas.nii.gz" \
      "${ATLAS_HOME}/${ATLAS_LB}" \
      "${QC_IMG_DIRNAME}/CT_Atlas_labels.png" \
      "CT->Atlas + labels"
  fi

# ========================================================================
# QC STEP 5B: CT->Atlas quantitative checks
#   1) Dice between transformed nnUNet brain mask and atlas brain mask
#   2) Similarity metrics inside atlas brain mask: 
#        MI between CT and atlas CT after full registration (Rigid + Affine + SyN).
# ========================================================================
  if is_true "$QC_ENABLED"; then


  # ---------------------------------------------------------
  # QC 5B.1: Brain mask Dice (nnUNet IA mask -> atlas space)
  # Requires atlas.brain_mask in config (ATLAS_BM)
  # ---------------------------------------------------------
   if [[ -n "$ATLAS_BM" && -f "${ATLAS_HOME}/${ATLAS_BM}" ]]; then
    log "QC 5B.1: Dice of brain mask in atlas space (NearestNeighbor)"

    "$ANTSAPPLY" \
      -d 3 \
      -i 02.brainMask_ia.nii.gz \
      -r "${ATLAS_HOME}/${ATLAS_CT}" \
      -t AT_1Warp.nii.gz \
      -t AT_0GenericAffine.mat \
      -n NearestNeighbor \
      -o brainmaskIA_atlas_space.nii.gz \
      2>&1 | tee "${POS_DIR}/logs/apply_mask_to_atlas.log"

     BMDICE="$(dice_safe brainmaskIA_atlas_space.nii.gz "${ATLAS_HOME}/${ATLAS_BM}")"
     echo "BrainMask_Dice_IA: ${BMDICE}" >> QC_metrics.txt
   else
     log "QC 5B.1: atlas brain_mask not provided -> Dice skipped"
   fi

  # ---------------------------------------------------------------------
  # QC 5B.2: MI inside atlas brain mask (MeasureImageSimilarity)
  # Requires MeasureImageSimilarity + atlas.brain_mask
  # ---------------------------------------------------------------------
    if [[ -n "$MEASURE_MI" && -x "$MEASURE_MI" && -n "$ATLAS_BM" && -f "${ATLAS_HOME}/${ATLAS_BM}" ]]; then
      log "QC 5B.2: MI between registered CT and atlas CT, restricted to atlas brain mask"
      
      FIXED_NATIVE="${ATLAS_HOME}/${ATLAS_CT}"
      MASK="${ATLAS_HOME}/${ATLAS_BM}"
      
      # Similarity metrics computed inside MASK on FIXED+MOVING
      CT_ATLAS_METRIC="$(mis_safe_mask "MI[${FIXED_NATIVE},out_CT2atlas.nii.gz,1,32]" "${MASK}" "${POS_DIR}/logs/qc_ct_atlas_mis_err.log")"
        
      {
        echo "---- QC CT->Atlas numeric ----"
        echo "CT_Atlas_METRIC:  ${CT_ATLAS_METRIC}"
      } >> QC_metrics.txt
    else
     log "QC 5B.2: MeasureImageSimilarity and/or atlas brain_mask missing -> MI skipped"
    fi

# ========================================================================
# QC STEP 5C: Append one line to global QC_summary.csv
# Columns: run_dir, position, suv_file, pet_ct_METRIC, brainmask_dice, ct_atlas_METRIC
# ========================================================================
    if is_true "$QC_WRITE_CSV"; then
      log "QC 5C: Appending QC row to QC_summary.csv"
      echo "${RUN_DIR},${i},${POS_DIR}/SUV.txt,${PET_CT_METRIC},${BMDICE},${CT_ATLAS_METRIC}" >> "$QC_SUMMARY"
    fi
  fi

  log "HOTEL_POS_${i} done."
  cd "$RUN_DIR"
done

log "Pipeline complete."