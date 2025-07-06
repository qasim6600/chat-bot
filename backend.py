from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pdfplumber
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from pyngrok import ngrok
import uvicorn

# ----- INIT -----
app = FastAPI()

# ----- Static Files -----
current_dir = os.path.dirname(os.path.realpath(__file__))

#app.mount("/static", StaticFiles(directory=".", html=True), name="static")
# ----- CORS -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- MODEL & CLIENT -----
model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key="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")

# ----- PDF UTILS -----
def clean_text(text):
    text = re.sub(r'Page\s*\d+\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\w)([A-Z]{2,})", r"\n\1", text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def is_inside(small_box, big_box):
    sx0, sy0, sx1, sy1 = small_box
    bx0, by0, bx1, by1 = big_box
    return (sx0 >= bx0 and sy0 >= by0 and sx1 <= bx1 and sy1 <= by1)

def extract_manual_content(pdf_path):
    normal_texts = []
    figure_texts = []
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            images = page.images
            words = page.extract_words()
            for word in words:
                word_bbox = (float(word['x0']), float(word['top']), float(word['x1']), float(word['bottom']))
                inside_image = any(is_inside(word_bbox, (float(img['x0']), float(img['top']), float(img['x1']), float(img['bottom']))) for img in images)
                if inside_image:
                    figure_texts.append(word['text'])
                else:
                    normal_texts.append(word['text'])
            page_tables = page.extract_tables()
            for table in page_tables:
                if table:
                    tables.append(table)
    normal_text = clean_text(" ".join(normal_texts))
    figure_text = clean_text(" ".join(figure_texts))
    return normal_text + "\n" + figure_text, tables

def chunk_text(text):
    heading_pattern = re.compile(r'^(?:[A-Z][A-Z\s]{3,}|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)$', re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    chunks = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = match.group().strip()
        body = text[start:end].strip()
        if body:
            chunks.append(f"{heading}\n{body}")
    return chunks

def chunk_tables(tables):
    table_chunks = []
    for table in tables:
        table_text = "\n".join([" | ".join([cell if cell else "" for cell in row]) for row in table])
        table_chunks.append(table_text)
    return table_chunks

def search_chunks(query, embeddings, chunks, top_k=2):
    query_embed = model.encode(query)
    similarities = cosine_similarity([query_embed], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def validate_query(query, manual_name):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a strict assistant validating user questions for the product manual titled '{manual_name}'.\n"
                    "If the question is relevant to installation, operation, troubleshooting, or safety of the product, respond with exactly: Valid Question.\n"
                    "If it is NOT relevant, you MUST respond in this exact format:\n"
                    "Invalid Input: Try questions like:\n"
                    "- [example 1]\n"
                    "- [example 2]\n"
                    "- [example 3]"
                )
            },
            {"role": "user", "content": query}
        ],
        temperature=0.4,
        max_tokens=120
    )
    result = response.choices[0].message.content.strip()
    if result.startswith("Valid Question"):
        return "Valid Question"
    elif result.startswith("Invalid Input:"):
        return result
    else:
        return "Invalid Input: Unexpected format."

def format_context_chunks(chunks):
    return "\n".join([f"- {chunk}" for chunk in chunks])

def extract_answer(query, text_context, table_context, product_name):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": f"""
You are a professional technical assistant for {product_name}.
You MUST answer using ONLY the provided CONTEXT.
- Do NOT mention or reference the context types.
- Combine details from both TEXT and TABLE context if needed.
- If answer is missing, say: "Sorry, I could not find this information in the manual."
"""}, 
            {"role": "user", "content": f"QUERY: {query}\n\nTEXT CONTEXT:\n{format_context_chunks(text_context)}\n\nTABLE CONTEXT:\n{format_context_chunks(table_context)}"}
        ],
        temperature=0,
        max_tokens=1200
    )
    return response.choices[0].message.content.strip()

# ----- Manuals -----
manuals = {
    "ElectroLux washing Machine": "E:/python/elctrolux.pdf"
}

manual_data = {}

for product_name, pdf_file in manuals.items():
    try:
        combined_text, tables = extract_manual_content(pdf_file)
        text_chunks = chunk_text(combined_text)
        table_chunks = chunk_tables(tables)
        text_embeddings = model.encode(text_chunks)
        table_embeddings = model.encode(table_chunks)
        manual_data[product_name] = {
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "text_embeddings": text_embeddings,
            "table_embeddings": table_embeddings
        }
        print(f"✅ Loaded {product_name}")
    except Exception as e:
        print(f"⚠️ Error loading {product_name}: {e}")

# ----- Widget API -----
class AskRequest(BaseModel):
    question: str
    product: str

class AskResponse(BaseModel):
    answer: str

from fastapi.responses import FileResponse

@app.get("/")
async def serve_root():
    return FileResponse("index.html")
@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if request.product not in manual_data:
        return {"answer": "⚠ Invalid product selected."}
    validation = validate_query(request.question, request.product)
    if validation != "Valid Question":
        return {"answer": validation}
    data = manual_data[request.product]
    top_text_chunks = search_chunks(request.question, data["text_embeddings"], data["text_chunks"])
    top_table_chunks = search_chunks(request.question, data["table_embeddings"], data["table_chunks"])
    answer = extract_answer(request.question, top_text_chunks, top_table_chunks, request.product)
    return {"answer": answer}

# ----- WhatsApp Webhook -----
@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    try:
        user_message = data['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']
        sender_id = data['entry'][0]['changes'][0]['value']['messages'][0]['from']
        product_name = "ElectroLux washing Machine"
        validation = validate_query(user_message, product_name)
        if validation != "Valid Question":
            answer = validation
        else:
            data = manual_data[product_name]
            top_text_chunks = search_chunks(user_message, data["text_embeddings"], data["text_chunks"])
            top_table_chunks = search_chunks(user_message, data["table_embeddings"], data["table_chunks"])
            answer = extract_answer(user_message, top_text_chunks, top_table_chunks, product_name)
        print(f"✅ WhatsApp [{sender_id}] : {answer}")
    except Exception as e:
        print(f"⚠️ WhatsApp Webhook Error: {e}")
    return {"status": "received"}

# ----- Instagram Webhook -----
@app.post("/webhook/instagram")
async def instagram_webhook(request: Request):
    data = await request.json()
    try:
        user_message = data['entry'][0]['messaging'][0]['message']['text']
        sender_id = data['entry'][0]['messaging'][0]['sender']['id']
        product_name = "ElectroLux washing Machine"
        validation = validate_query(user_message, product_name)
        if validation != "Valid Question":
            answer = validation
        else:
            data = manual_data[product_name]
            top_text_chunks = search_chunks(user_message, data["text_embeddings"], data["text_chunks"])
            top_table_chunks = search_chunks(user_message, data["table_embeddings"], data["table_chunks"])
            answer = extract_answer(user_message, top_text_chunks, top_table_chunks, product_name)
        print(f"✅ Instagram [{sender_id}] : {answer}")
    except Exception as e:
        print(f"⚠️ Instagram Webhook Error: {e}")
    return {"status": "received"}

# ----- Start -----
if __name__ == "__main__":
    public_url = ngrok.connect(8000)
    print(f"🚀 Public URL: {public_url}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
