from pathlib import Path
import os
import shutil
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tomllib
import markdown
import uvicorn

# TODO Organize this

# REVIEW consider renaming to something not called main
from main import RAGPipeline

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

rag = RAGPipeline()
# rag.build_knowledge_base()
# rag.process_data_sources()

CONFIG_PATH = "config.toml"

# Open configuration
with Path(CONFIG_PATH).open("rb") as f:
    config_data = tomllib.load(f)

AUDIO_FOLDER = config_data.get("audio_folder", "input-data")
DOCS_FOLDER = config_data.get("docs_folder", "input-data")
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(DOCS_FOLDER, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_FOLDER), name="audio")
app.mount("/docs", StaticFiles(directory=DOCS_FOLDER), name="docs")


templates = Jinja2Templates(directory="templates")


# Home page
@app.get("/", response_class=HTMLResponse)
def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Chat page
@app.get("/chat", response_class=HTMLResponse)
def get_chat(request: Request):
    return templates.TemplateResponse("fragments/chat.html", {"request": request})


# Ask endpoint
@app.post("/ask", response_class=HTMLResponse)
def post_ask(request: Request, query: str = Form(...)):
    raw_answer = rag.query(query)
    html_answer = markdown.markdown(raw_answer, extensions=["fenced_code", "tables"])

    return templates.TemplateResponse(
        "fragments/answer.html", {"request": request, "answer": html_answer}
    )


# Config page
@app.get("/config", response_class=HTMLResponse)
def get_config(request: Request):
    try:
        with open(CONFIG_PATH, "r") as f:
            config_content = f.read()
    except FileNotFoundError:
        config_content = "# config.toml not found\n"

    return templates.TemplateResponse(
        "fragments/config.html",
        {"request": request, "config_content": config_content},
    )


# Save config endpoint
@app.post("/save-config", response_class=HTMLResponse)
def post_save_config(config: str = Form(...)):
    with open(CONFIG_PATH, "w") as f:
        f.write(config)
    return HTMLResponse("<p>Config saved ✅</p>")


# Input data page
@app.get("/data", response_class=HTMLResponse)
def get_data_page(request: Request):
    audio_files = os.listdir(AUDIO_FOLDER)
    doc_files = os.listdir(DOCS_FOLDER)
    return templates.TemplateResponse(
        "fragments/data.html",
        {"request": request, "audio_files": audio_files, "doc_files": doc_files},
    )


@app.post("/upload-audio", response_class=HTMLResponse)
async def upload_audio(file: UploadFile = File(...)):
    path = os.path.join(AUDIO_FOLDER, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return HTMLResponse(f"<p>Uploaded to audio folder: {file.filename} ✅</p>")


@app.post("/upload-doc", response_class=HTMLResponse)
async def upload_doc(file: UploadFile = File(...)):
    path = os.path.join(DOC_FOLDER, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return HTMLResponse(f"<p>Uploaded to document folder: {file.filename} ✅</p>")


# Refresh knowledge base
@app.post("/refresh-kb", response_class=HTMLResponse)
def refresh_knowledge_base():
    rag.build_knowledge_base()
    return HTMLResponse("<p>Knowledge base updated ✅</p>")


# Process data input
@app.post("/process-data", response_class=HTMLResponse)
def process_data_input():
    rag.process_data_sources()
    return HTMLResponse("<p>Processed data sources ✅</p>")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
