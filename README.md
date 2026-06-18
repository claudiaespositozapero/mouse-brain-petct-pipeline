# Automated CT-based Pipeline for Mouse Brain PET Quantification

A fully automated, open-source pipeline for atlas-normalized mouse brain PET-CT quantification. No subject-specific MRI required. No manual intervention needed.

> **Associated publication:**  
> Esposito-Zapero C, Acebal S, Padro D, Arsequell G, Aguiar P, Martín A, Galdran A, Llop J.  
> *An open and fully automated CT-based pipeline for high-throughput mouse brain PET quantification.*  
> *(Journal – under review)*

---

## Overview

This pipeline processes mouse brain PET-CT data from raw DICOM files to atlas-based regional SUV values in a single automated run. It supports both single-animal and multi-animal ("mouse-hotel") acquisition configurations.

**Key features:**
- DICOM-to-NIfTI conversion (dcm2niix)
- SUV scaling
- Automatic field-of-view splitting for multi-animal beds (1, 3, or 4 animals)
- Rigid PET-to-CT registration (ANTs)
- Deep learning CT brain extraction (nnU-Net)
- Multi-stage deformable CT-to-atlas registration (ANTs: Rigid + Affine + SyN)
- Atlas-based regional SUV extraction (Convert3D)
- Automated QC outputs: visual overlays + quantitative metrics (Dice, MI CT→Atlas, MI PET→CT)

---

## Requirements

### Software dependencies

| Tool | Version tested | Purpose |
|------|---------------|---------|
| ANTs | 2.6.5 | Image registration |
| nnU-Net v2 | 2.6.4 | CT brain segmentation |
| Convert3D (c3d) | 1.1.0 | Image manipulation, SUV scaling, ROI stats |
| dcm2niix | 1.0.20220720 | DICOM to NIfTI conversion |
| FSL (fslstats) | 6.0.7.19 | Optional voxel count QC |
| Python | ≥ 3.9 | QC image generation |
| Python packages | — | numpy, nibabel, matplotlib, pyyaml |

Install Python dependencies:
```bash
pip install numpy nibabel matplotlib pyyaml
```

### Installing nnU-Net:
We recommend installing nnU-Net v2 directly from the official repository, as installation steps vary depending on your operating system, Python environment, and hardware. Detailed instructions for all configurations are available at:

- General installation: https://github.com/MIC-DKFZ/nnUNet
- Step-by-step setup guide: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/getting-started/installation-and-setup.md

### Installing system tools (ANTs, Convert3D, dcm2niix)

⚠️ ANTs, Convert3D (c3d), and dcm2niix are system-level tools and **cannot be installed via pip**. Installation depends on your operating system:

**ANTs:**
```bash
sudo apt install ants          # Ubuntu/Debian
```
Or download pre-compiled binaries from: https://github.com/ANTsX/ANTs/releases

**Convert3D (c3d):**
```bash
sudo apt install python3-dev   # Ubuntu/Debian (required dependency)
```
Then download the c3d binary from: https://sourceforge.net/projects/c3d/

**dcm2niix:**
```bash
# Ubuntu/Debian — see full instructions at:
# https://github.com/rordenlab/dcm2niix#install
sudo apt install dcm2niix
```

**FSL** (optional, only needed for voxel count QC):
See installation instructions at: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

### Atlas and model files

- **Atlas:** T2-weighted mouse brain MRI template (Mirrione et al., 2007) with atlas label map
- **nnU-Net brain extraction model:** trained on 100 rodent CT volumes (50 mice, 50 rats)

> Please contact the authors or refer to the publication for access to the atlas and the pretrained nnU-Net model.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/petct-brain-pipeline.git
cd petct-brain-pipeline
chmod +x petct_coreg_atlas.sh
chmod +x batch_run.sh
```

No compilation is needed. The pipeline is a self-contained Bash script.

---

## Input data organization

Each acquisition must be stored in its own directory containing:

```
my_acquisition/
├── *PET*.dcm          # All PET DICOM files
├── *CT*.dcm           # All CT DICOM files
├── SUV.txt            # Single animal: one SUV scaling factor
│                      # Multi-animal: SUV_1.txt, SUV_2.txt, ... SUV_4.txt
└── config.yaml        # Configuration file (see below)
```

**SUV file format:** plain text file with a single numeric value (e.g., `0.1234`), representing the multiplicative factor to convert PET voxel intensities to SUV units.

⚠️ **Image orientation:** The pipeline was developed and validated using images in RPS orientation, as produced by the Molecubes β-Cube/X-Cube system. Images
acquired on other scanners should be verified and reoriented if necessary prior to processing. For multi-animal acquisitions, correct orientation is particularly
important as the geometric FOV splitting assumes a fixed spatial arrangement of animals along the X and Y axes.

---

## Configuration file

Copy and edit the provided template:

```bash
cp config_template.yaml config.yaml
```

Key fields to set:

```yaml
# Paths to atlas files
atlas:
  home: /path/to/atlas/directory
  ct: atlas_ct.nii.gz
  labels: atlas_labels.nii.gz
  brain_mask: atlas_brain_mask.nii.gz   # optional, needed for Dice QC

# Paths to tools (leave as-is if tools are on your PATH)
tools:
  dcm2niix: dcm2niix
  c3d: c3d
  antsRegistration: antsRegistration
  antsApplyTransforms: antsApplyTransforms
  ImageMath: ImageMath
  MeasureImageSimilarity: MeasureImageSimilarity  # optional
  fslstats: fslstats                              # optional
  nnUNetv2_predict: nnUNetv2_predict

# nnU-Net model settings
nnunet:
  results_dir: /path/to/nnunet/results
  dataset: Dataset001_BrainCT
  configuration: 2d
  trainer: nnUNetTrainer
  plans: nnUNetPlans
  device: cpu          # or cuda
  folds: "0 1 2 3 4"

# Output directory
output:
  dir: ./output

# Quality control outputs
qc:
  enabled: true
  write_csv: true
  write_images: true
```

---

## Usage

### Single acquisition

```bash
cd /path/to/my_acquisition/
/path/to/petct_coreg_atlas.sh --config config.yaml --hotel {1|3|4}
```

**`--hotel` values:**
- `1` — single-animal bed
- `3` — three-animal mouse-hotel
- `4` — four-animal mouse-hotel

**Example:**
```bash
cd /data/mouse01/
../petct_coreg_atlas.sh --config config.yaml --hotel 1
```

### Batch processing

To process a set of acquisitions sequentially, use the companion script `batch_run.sh`.
Each subdirectory in the batch folder is processed as an independent run using the same
`--hotel` value, so keep single-animal and hotel acquisitions in separate batch folders.

```
my_batch_folder/
├── scan_01/    # *PET*.dcm, *CT*.dcm, SUV.txt
├── scan_02/
├── scan_03/
└── ...
```

```bash
bash batch_run.sh \
  --pipeline /path/to/petct_coreg_atlas.sh \
  --config   config.yaml \
  --batch    /path/to/my_batch_folder/ \
  --hotel    1
```

Results for each scan are written to `<subdir>/output/`. A per-run log is saved to
`<subdir>/output/pipeline.log`. If QC is enabled, a consolidated QC table across all
runs is written to `<batch_dir>/batch_QC_summary.csv`.

If a run fails, the batch continues processing the remaining acquisitions and reports
a summary of failed and skipped runs at the end.

---

## Output structure

```
output/
├── cropped/
│   ├── 01.CT_POS_1.nii.gz
│   └── 01.PET_POS_1.nii.gz
├── HOTEL_POS_1/
│   ├── 01.PETSUV.nii.gz           # PET in SUV units
│   ├── 01.SUV2CT.nii.gz           # PET registered to CT
│   ├── 02.brainMask_ia.nii.gz     # nnU-Net brain mask
│   ├── 02.brainMask.nii.gz        # Dilated brain mask
│   ├── 03.brainCT_trim.nii.gz     # Brain-extracted CT
│   ├── out_CT2atlas.nii.gz        # CT normalized to atlas space
│   ├── out_SUV2Atlas.nii.gz       # PET in atlas space
│   ├── SUV_values.csv             # ← Main quantitative output
│   ├── QC_metrics.txt             # Per-position QC summary
│   └── qc_images/
│       ├── PET_CT_overlay.png
│       └── CT_Atlas_labels.png
└── QC_summary.csv                 # Global QC table (all positions)

logs/
├── dcm2niix_pet.log
└── dcm2niix_ct.log
```

**`SUV_values.csv`** contains: Label ID, Mean SUV, Number of voxels — one row per atlas region.

---

## Quality control metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Dice** | Overlap between warped nnU-Net mask and atlas brain mask | 0–1; values >0.90 indicate good registration |
| **MI CT→Atlas** | Mutual information inside atlas brain mask after full registration | More negative = better; use as within-group consistency indicator |
| **MI PET→CT** | Mutual information between CT and registered PET | More negative = better; values depend on tracer and acquisition mode |

> ⚠️ Low Dice values (e.g., <0.85) may indicate CT acquisition artefacts or segmentation failures. Visual inspection of `qc_images/` is recommended for flagged scans.

---

## Multi-animal bed positions

For hotel acquisitions, animals must be placed at predefined bed positions. The FOV is partitioned using fixed geometric splits:

| Position | XY origin | XY size |
|----------|-----------|---------|
| POS_1 | 50%, 50% | 50%, 50% |
| POS_2 | 0%, 50% | 50%, 50% |
| POS_3 | 0%, 0% | 50%, 50% |
| POS_4 | 50%, 0% | 50%, 50% |

Each position is processed independently. Positions without a corresponding `SUV_i.txt` file are skipped.

---

## Validation

The pipeline was validated against manual PMOD-based analysis in >600 PET acquisitions:

- **Tracers:** [¹⁸F]Florbetaben, florzolotau (¹⁸F), [¹⁸F]DPA-714
- **Regions:** hippocampus, cortex, cerebellum, thalamus
- **Configurations:** single-animal and 4-animal mouse-hotel
- **Agreement (single-animal):** r = 0.907–0.997 (all p < 0.0001)
- **Agreement (mouse-hotel):** r = 0.868–0.981 (all p < 0.0001)

---

## Citation

If you use this pipeline in your research, please cite:

> Esposito-Zapero C, Acebal S, Padro D, Arsequell G, Aguiar P, Martín A, Galdran A, Llop J.  
> *An open and fully automated CT-based pipeline for high-throughput mouse brain PET quantification.*  
> *(Citation to be updated upon publication)*

**Atlas reference:**  
> Ma Y, Hof PR, Grant SC, et al. A three-dimensional digital atlas database of the adult C57BL/6J mouse brain by magnetic resonance microscopy. *Neuroscience.* 2005;135. https://doi.org/10.1016/j.neuroscience.2005.07.014

---

## License

This software is released for academic use. See [LICENSE](LICENSE) for details.

---

## Contact

For questions or issues, please open a [GitHub Issue](../../issues) or contact the corresponding author:  
**Jordi Llop** — jllop@cicbiomagune.es  
CIC BiomaGUNE, Basque Research and Technology Alliance (BRTA), Donostia-San Sebastián, Spain.
