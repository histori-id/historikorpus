# HistoriKorpus
**HistoriKorpus** is the first dataset of OCR output and ground truth for historical Indonesian newspapers published in the post-independence era (1946–1965). The dataset is designed to support the development and evaluation of OCR and post-OCR correction models for historical Indonesian texts, as well as broader downstream NLP tasks such as named entity recognition, text summarization, and sentiment analysis.ditranskrip.

## Dataset Overview

| | |
|---|---|
| **Articles** | 1,489 manually transcribed articles |
| **Parallel excerpts** | 5,290 paired OCR and ground truth excerpts |
| **Newspaper pages** | 101 pages from 6 newspaper titles |
| **Period covered** | 1946–1965 |
| **Total tokens** | ~212k tokens (~1,365k characters) |
| **Baseline CER** | 8.25% |
| **Baseline WER** | 34.22% |

## Newspapers Included

| Newspaper | Year | Pages |
|---|---|---|
| Soeara Oemoem | 1946–1947 | 25 |
| Kedaulatan Rakjat | 1950–1958 | 26 |
| Duta Pantjasila | 1959 | 6 |
| Indonesia Merdeka | 1960 | 20 |
| Bintang Timoer | 1963 | 10 |
| Duta Masjarakat | 1963–1965 | 16 |

## Data Access

The full dataset is publicly available on Mendeley Data:

> Afriyanti, I., Muhammad Azzam, N. (2026). *HistoriKorpus*. 
> Mendeley Data. https://doi.org/10.17632/b6p3jgdx83.1

## Data Source

The newspaper scans used in this dataset were sourced from the 
[Khastara Repository](https://khastara.perpusnas.go.id/), the digital 
archive of the National Library of Indonesia (Perpusnas). A subset of 
101 pages was selected for OCR processing and manual transcription. 
The original scans used in this work are accessible via the following 
link:

📄 [Download Newspaper Scans (Google Drive)](https://drive.google.com/drive/folders/1yJhS--l3nBGNG5S0bXbqtbBmGj_A9rg7?usp=sharing)

> **Note:** The scans are provided solely for research and academic 
> purposes. All original materials remain the property of the National 
> Library of Indonesia (Perpusnas).

## Dependencies
```bash
!pip install doclayout-yolo
!sudo apt install tesseract-ocr
!pip install pytesseract
!apt-get install -y tesseract-ocr
!apt-get install -y libtesseract-dev
!sudo apt install tesseract-ocr-ind
!sudo apt install tesseract-ocr-nld
```

## 📜 License & Citation

This repository uses a dual-licensing scheme:

- **Transcriptions & OCR Data:** Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the text data as long as appropriate credit/citation is given.
- **Source Newspaper Images:** Marked as [Public Domain Mark 1.0 (PDM 1.0)](https://creativecommons.org/publicdomain/mark/1.0/) due to the expiration of copyright on the original historical publications.

### How to Cite
If you use this dataset in your research, please cite our work:
```bibtex
@article{your_paper_2026,
  title={Title of Your Paper/Dataset},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2026}
}

```markdown
[![License: CC BY 4.0](https://img.shields.io/badge/Data_License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: PDM 1.0](https://img.shields.io/badge/Images_License-PDM_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/mark/1.0/)

## Acknowledgements

This research was supported and funded by Fakultas Ilmu Komputer Universitas Indonesia under Grant Agreement 
No. NKB-3/UN2.F11.D/HKP.05.00/2025.
