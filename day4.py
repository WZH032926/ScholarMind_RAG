import json
import os
from PyPDF2 import PdfReader, PdfFileReader
from tqdm import tqdm
from openai import OpenAI

PDF_PATH = "../学习存档/大模型电价预测.pdf"
JSON_SAVE_PATH = "../学习存档/knowledge_base.json"

API_KEY = "f7faed3eb36840468e29b38d5de3884c.Y0Ce9FmrUCsFrcea"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
EMBEDDING_MODEL = "embedding-2"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_text_from_pdf(filename):
    print(f"正在解析PDF：{filename}")
    #先处理异常：工程化思维
    if not os.path.exists(filename):
        raise FileNotFoundError(f"找不到文件：{filename}，请确认文件路径！")

    text = ""
    with open(filename, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    print(f"解析完成，共{len(text)} 个字符。")
    return text

def split_text_with_overlap(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    total = len(text)

    while start < total:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += (chunk_size - overlap)
    print(f"切分完成，共生成 {len(chunks)} 个文本块。")
    return chunks

def get_embedding(text):
    clean_text = text.replace("\n", " ")

    try:
        response = client.embeddings.create(
            input=clean_text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding 失败: {e}")
        return []

if __name__ == "__main__":
#提取本文
    raw_text = extract_text_from_pdf(PDF_PATH)
#切分文本
    text_chunks = split_text_with_overlap(raw_text, chunk_size=500, overlap=50)

#向量化
    knowledge_base = []

    print("开始向量化，耐心等待....")

    # tqdm(text_chunks) 会自动显示进度条，让你知道处理到第几个了
    for i, chunk in enumerate(tqdm(text_chunks)):
        vector = get_embedding(chunk)

        if vector:
            record = {
                "id": i,
                "text": chunk,
                "vector": vector,
                "source": PDF_PATH
            }
            knowledge_base.append(record)
    print(f"正在保存数据到{JSON_SAVE_PATH}.....")
    with open(JSON_SAVE_PATH, "w",  encoding='utf-8') as f:
        # ensure_ascii=False 保证中文能正常显示，不会变成 \uXXXX
        # indent=4 让文件排版美观，不是挤在一行
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)

    print("ETL 流程结束！ 请查看JSON文件。")






























