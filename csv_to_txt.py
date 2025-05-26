import csv
import re

input = "data/random/random_origin.txt"
output_txt = "data/random/random.txt" 

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9,.!?;:'\"()\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

with open(input, newline='', encoding='utf-8') as csvfile, \
     open(output_txt, 'w', encoding='utf-8') as outfile:
    reader = csv.reader(csvfile)
    header = next(reader)

    for row in reader:
        text = " ".join(row)
        clean = clean_text(text)
        # 简单分词：按空格分开，重新用空格连接，保持一致格式
        tokens = clean.split()
        outfile.write(" ".join(tokens) + "\n")

print(f"Prepocessing finished:{output_txt}")