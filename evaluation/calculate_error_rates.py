import json
import string
from jiwer import wer, cer

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return ' '.join(text.split()).strip()

def process_json(json_path: str):
    concatenated_reference = ""
    concatenated_hypothesis = ""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_items = len(data)
    print(f"Found {total_items} items")

    for i, item in enumerate(data, start=1):
        reference_text = item.get("transcription_gt", "")
        hypothesis_text = item.get("ocr_raw", "")

        concatenated_reference += reference_text + " "
        concatenated_hypothesis += hypothesis_text + " "

        print(f"{i} item(s) processed")

    # Normalize texts
    ref = normalize_text(concatenated_reference)
    hyp = normalize_text(concatenated_hypothesis)

    # Compute WER and CER
    total_wer_score = wer(ref, hyp)
    total_cer_score = cer(ref, hyp)

    return total_wer_score, total_cer_score

# Path to JSON file
json_path = "/content/output.json"

# Run and print
total_wer, total_cer = process_json(json_path)

print(f"Total WER: {total_wer:.5f}")
print(f"Total CER: {total_cer:.5f}")
