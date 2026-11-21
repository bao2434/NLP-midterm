### 1. OCR
To run in Kaggle, do the following:
- Zip NLP-data and NLP-module.
- Create a Kaggle dataset for each ZIP file and upload it.
- Upload nlp-ocr.ipynb to Kaggle.
- Inspect the folder structure and update the notebook paths accordingly.
- Enable GPU.
- Run automatically using **Save Version**.

### 2. Preprocessing
To run in Kaggle, do the following:
- Upload the Han and Vietnamese TXT files and HanVietPreprocessor.py to Kaggle.
- Upload 3_tien_xu_ly_han_viet.ipynb.
- Inspect the input paths and update the notebook accordingly.
- Run using **Save Version** (CPU is sufficient).
- **Note:** The current preprocessing pipeline is rule-based and does not use an LLM.

### 3. Sentence Alignment
To run in Kaggle, do the following:
- Upload the Han and Vietnamese TXT files and the alignment modules.
- Upload 3_dong_hang_voting_nu_khoa.ipynb.
- Inspect the input paths and update the notebook accordingly.
- Enable Internet (GPU is optional).
- Run Bertalign, Vecalign, and CLC API with majority voting using **Save Version**.
- **Note:** Automated error detection and handling for notes.xlsx will be implemented later.