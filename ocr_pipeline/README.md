Panduan untuk running OCR pipeline
Install 
pip install -r requirements.txt
pip install -r apt-packages.txt 

Lakukan preprocessing
Initial statenya adalah dengan mendownload dokumen .pdf scanned dari Perpusnas, kemudian displit per page kemudian diubah ke JPG dengan 300DPI.
File JPG disimpan dalam folder jpg kemudian run kode run_ocr.py

Output akan disimpan di folder ocr-result
Jelasin output folder di sini:

Algoritma kode:
- Algoritma merging rules
