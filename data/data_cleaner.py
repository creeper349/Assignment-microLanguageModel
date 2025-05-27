from Bio import Entrez
import re
import time

Entrez.email = "1601789895@qq.com"  

def clean_abstract(text):
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-', ' ', text)              
    text = re.sub(r'\[\d+\]', '', text)          
    text = re.sub(r'\(\d+\)', '', text)           
    text = re.sub(r'\d+\.', '', text)             
    text = re.sub(r'([.,!?;:()"\'])', r' \1 ', text)  
    text = re.sub(r'\s+', ' ', text)              
    return text.strip()

def fetch_clean_pubmed_abstracts(query, max_results=10000):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()

    id_list = record["IdList"]
    abstracts = []

    batch_size = 20
    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i:i+batch_size]
        fetch = Entrez.efetch(db="pubmed", id=",".join(batch_ids),
                              rettype="medline", retmode="text")
        raw = fetch.read()
        fetch.close()

        for abs_match in re.findall(r"AB  - (.*?)\n(?:\w\w  - |\n\n)", raw, flags=re.DOTALL):
            abstract = abs_match.replace("\n", " ")
            cleaned = clean_abstract(abstract)
            if cleaned:
                abstracts.append(cleaned)

        time.sleep(0.4)

    return abstracts

query = "metabolism"
abstracts = fetch_clean_pubmed_abstracts(query, max_results=10000)

with open("/home/stu13/Language_model/data/biochem/valid.txt", "w", encoding="utf-8") as f:
    for abs in abstracts:
        f.write(abs + "\n")

print(f"共保存 {len(abstracts)} 条摘要。")