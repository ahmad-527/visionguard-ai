# MVTec AD 2 local setup and audit

MVTec AD 2 is the primary VisionGuard benchmark. Dataset files are deliberately
not included in this repository. The dataset is published by MVTec under
CC BY-NC-SA 4.0 for non-commercial use; review the current terms yourself before
obtaining or using it.

## Obtain the dataset

1. Visit the [official MVTec AD 2 page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2).
2. Read the license terms and complete MVTec's download form.
3. Download and extract the dataset outside this repository. Do not place the
   archive, extracted images, or annotations in Git.
4. Keep the official category directories directly beneath a single local root.

The checked-in dataset contract at
`configs/datasets/mvtec_ad_2.yaml` follows MVTec's public dataset utility:

```text
<dataset-root>/
  can/
    train/good/*.png
    validation/good/*.png
    test_public/good/*.png
    test_public/bad/*.png
    test_public/ground_truth/bad/*_mask.png
    test_private/*.png
    test_private_mixed/*.png
  fabric/
  fruit_jelly/
  rice/
  sheet_metal/
  vial/
  wallplugs/
  walnuts/
```

The audit does not enforce hard-coded file or category counts. Counts in its
report are measurements of the supplied filesystem. If MVTec changes the
release layout, update the dataset configuration only after verifying the new
official structure.

## Configure the local path

Pass the dataset root on each command. This avoids committing a developer's
machine-specific path or storing it in source configuration. A shell variable is
convenient but optional:

```powershell
$VisionGuardDataset = "D:\datasets\mvtec_ad_2"
visionguard-audit $VisionGuardDataset `
  --config configs/datasets/mvtec_ad_2.yaml `
  --output audit-reports/mvtec-ad-2.json
```

On macOS or Linux, the equivalent is:

```bash
VISIONGUARD_DATASET=/datasets/mvtec_ad_2
visionguard-audit "$VISIONGUARD_DATASET" \
  --config configs/datasets/mvtec_ad_2.yaml \
  --output audit-reports/mvtec-ad-2.json
```

`audit-reports/` is ignored by Git. The command returns non-zero when errors are
found, after writing the report. Use `--fail-on warning` for stricter automation
or `--fail-on never` for exploratory inspection.

## What the audit verifies

- configured categories, splits, condition directories, and unexpected folders;
- actual image decodability, dimensions, color mode, and SHA-256 digest;
- public anomalous-image to segmentation-mask naming and dimensions;
- missing and orphan masks;
- duplicate content within a split;
- identical content crossing split boundaries, reported as leakage errors;
- actual counts by split and category in deterministic JSON.

The tool is read-only with respect to the dataset. It does not validate the
private labels (which are unavailable), download data, assess license compliance,
or establish scientific benchmark validity. Hash equality detects exact duplicate
bytes; it does not detect visually near-duplicate images.
