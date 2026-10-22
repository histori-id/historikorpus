[![License: CC BY 4.0](https://img.shields.io/badge/Data_License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: PDM 1.0](https://img.shields.io/badge/Images_License-PDM_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/mark/1.0/)

# HistoriKorpus
**HistoriKorpus** is the first dataset of OCR output and ground truth for historical Indonesian newspapers published in the post-independence era (1946–1965). The dataset is designed to support the development and evaluation of OCR and post-OCR correction models for historical Indonesian texts, as well as broader downstream NLP tasks such as named entity recognition, text summarization, and sentiment analysis. Although the initial objective was to construct an article-level corpus, severe text overlapping issues—where parts of one article inadvertently merged with another—led us to publish a parallel corpus at the block level. Nevertheless, the metadata for each text block is preserved within the dataset. Driven by this original intention to achieve article-level granularity, the current OCR pipeline available in the repository incorporates a grouping mechanism to assemble text blocks back into logical article structures.

## Dataset Overview

| | |
|---|---|
| **Articles** | 1,508 manually transcribed articles |
| **Parallel excerpts** | 5,383 paired OCR and ground truth excerpts |
| **Newspaper pages** | 100 pages from 6 newspaper titles |
| **Period covered** | 1946–1965 |
| **Total tokens** | ~285k tokens (~1,944k characters) |
| **Baseline CER** | 8.25% |
| **Baseline WER** | 34.22% |

## Newspapers Included

| Newspaper | Year | Pages |
|---|---|---|
| Soeara Oemoem | 1946–1947 | 25 |
| Kedaulatan Rakjat | 1950–1958 | 23 |
| Duta Pantjasila | 1959 | 6 |
| Indonesia Merdeka | 1960 | 20 |
| Bintang Timur | 1963 | 10 |
| Duta Masjarakat | 1963–1965 | 16 |

## Data Access

The full dataset is publicly available on Mendeley Data:
> Afriyanti, I., Muhammad Azzam, N. (2026). *HistoriKorpus*. 
> Mendeley Data. https://doi.org/10.17632/b6p3jgdx83.2

## Data Sources & Repository Access

The historical newspaper scans used in this dataset were sourced from the [Khastara Repository](https://khastara.perpusnas.go.id/), the digital archive of the National Library of Indonesia (Perpusnas). A curated subset of 100 pages was selected for OCR processing and manual transcription.

The original scans used in this work are accessible via [Mendeley Data](https://doi.org/10.17632/b6p3jgdx83.2). Alternatively, you can download the dataset directly via Google Drive:
- **PDF version:** [Google Drive Folder (PDF)](https://drive.google.com/drive/folders/1nHvsaHHgv_irN-P_Zcl0ypLloNWs7ADu?usp=drive_link)
- **JPG version:** [Google Drive Folder (JPG)](https://drive.google.com/drive/folders/1kuyo9A5_PLFx1juQpBx-LI6N176O3vtH?usp=drive_link)

## Reproducing the OCR
See [`ocr_pipeline/README.md`](ocr_pipeline/README.md). Re-running the pipeline with different hardware or library versions
may slightly change block boundaries and grouping.

## Dataset Splits for Post-OCR Correction

For **post-OCR error correction**, researchers can split the dataset at the excerpt level (using the paired records in `gt_ocr_excerpts`) into standard benchmark partitions. 

We recommend the following standard split ratio to ensure that text degradation characteristics and historical orthographic variations are balanced across all subsets:

* **Training Set (80%):** Used for training the error-correction model.
* **Validation Set (10%):** Used for hyperparameter tuning and monitoring performance during training.
* **Testing Set (10%):** Used for final objective evaluation (e.g., calculating final Character Error Rate / Word Error Rate reductions).

## 📜 License

This repository uses a dual-licensing scheme:

- **Transcriptions & OCR Data:** Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the text data as long as appropriate credit/citation is given.
- **Source Newspaper PDFs and Images:** Marked as [Public Domain Mark 1.0 (PDM 1.0)](https://creativecommons.org/publicdomain/mark/1.0/)


## Acknowledgements

This research was supported and funded by Fakultas Ilmu Komputer Universitas Indonesia under Grant Agreement 
No. NKB-3/UN2.F11.D/HKP.05.00/2025.
