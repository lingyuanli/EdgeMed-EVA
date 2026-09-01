# External Medical VQA Data Source Record

Checked: 2026-09-01. This file records source facts used by the external-data
decision. It does not treat dataset-card claims as experimental results.

## PMC-VQA v2

- Official dataset page: <https://huggingface.co/datasets/RadGenome/PMC-VQA>
- Frozen Hugging Face revision: `b56ae594f794867893143b337b4118a835794647`
- Upstream corpus policy: <https://pmc.ncbi.nlm.nih.gov/tools/openftlist/>
- Dataset card states that source articles came from the PMC commercial-use
  subset and that the resulting dataset is CC BY-SA. Article-level licenses
  must still be joined from `oa_comm_use_file_list.csv`.
- The Hugging Face viewer currently fails because v1 and v2 CSV schemas differ.
  The build downloads named files at the frozen revision instead of invoking a
  generic dataset loader.

| File | Bytes | SHA-256 / LFS oid |
|---|---:|---|
| `train_2.csv` | 56,703,454 | `15ead4a27b5365d1b5dc5faf1fd2246a6a5aa1ca5c2358a19532ff78c2870084` |
| `test_2.csv` | 12,446,345 | `d57d567f997955e7001ec4325323a2bca66d570070466291d5ec12882ff5ba09` |
| `images_2.zip` | 2,206,255,503 | `727643d0ae9182cb5572b43a74ed4100eb0920b1d83e55ed4054a393d672cb4a` |
| `oa_comm_use_file_list.csv` | 496,021,536 | `7a56eef6527332d2eb47fffa1356ffe19548b8984823a533d174fd70bb38b86f` |

`train_2.csv` was inspected directly. Its columns are `index`, `Figure_path`,
`Caption`, `Question`, four choices, `Answer`, and `split`. The questions and
answers are machine-generated from article figures/captions, so admitted
records must be labelled `annotation_type=synthetic`, never `human`.

Direct comparison of the frozen CSVs found 152,603 train rows over 135,339
images/62,503 PMC articles and 33,430 test rows over 29,021 images/11,112 PMC
articles. Exact figure-name overlap and PMCID overlap were both zero. Cycle 1
therefore selects training seeds only from train and a 512-record MCQ development
cohort only from test, with different frozen seeds and at most one question per
image.

## SLAKE

- Official dataset page: <https://huggingface.co/datasets/BoKelvin/SLAKE>
- Frozen Hugging Face revision: `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`
- Dataset-card license: CC BY 4.0.

| File | Bytes | SHA-256 / LFS oid |
|---|---:|---|
| `validation.json` | 639,139 | `32b016440b0c3be11056a78a18eeab46333268407fbb6e6b32f9f4c2debc50f6` |
| `imgs.zip` | 212,343,373 | `44eb7d9214e1ac5b7946e237b669401866df9114fd9a3618c5a84fbffcded0b0` |

Direct inspection of the frozen validation JSON found 2,099 questions: 1,053
English and 1,046 Chinese questions over 174 image names. The first optimization
cycle uses only the English official validation records. It preserves the
official split and groups all analysis by image, because multiple questions can
refer to the same image.

## Deferred evidence-localization sources

- MS-CXR v1.1.0: <https://physionet.org/content/ms-cxr/1.1.0/>. It contains
  expert phrase/box annotations, but access requires credentialing, training,
  a data-use agreement, and MIMIC-CXR-JPG access.
- VinDr-CXR v1.0.0: <https://physionet.org/content/vindr-cxr/1.0.0/>. It provides
  radiologist bounding boxes but is also credentialed.

These are later localization-evaluator candidates. They are not dependencies
of cycle 1 and must not be silently replaced with mirrors of unknown provenance.
