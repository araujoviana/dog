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

# TODO Rename the file
from main import RAGPipeline

# ======================
# Constants / Configuration
# ======================
CONFIG_PATH = "config.toml"

# Load configuration
with Path(CONFIG_PATH).open("rb") as f:
    config_data = tomllib.load(f)

AUDIO_FOLDER = config_data.get("audio_folder", "input-data")
DOCS_FOLDER = config_data.get("docs_folder", "input-data")
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(DOCS_FOLDER, exist_ok=True)

# ======================
# App Initialization
# ======================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/audio", StaticFiles(directory=AUDIO_FOLDER), name="audio")
app.mount("/docs", StaticFiles(directory=DOCS_FOLDER), name="docs")

templates = Jinja2Templates(directory="templates")
rag = RAGPipeline()


# ======================
# Helper Functions
# ======================
def markdown_chunks(chunks: list[str]) -> list[str]:
    """Convert a list of markdown strings to HTML."""
    return [markdown.markdown(c, extensions=["fenced_code", "tables"]) for c in chunks]


def save_file(folder: str, file: UploadFile) -> str:
    """Save uploaded file to specified folder."""
    path = os.path.join(folder, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return file.filename


# ======================
# Page Routes
# ======================
@app.get("/", response_class=HTMLResponse)
def get_home(request: Request):
    """Welcome page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
def get_chat(request: Request):
    """Chat interface."""
    return templates.TemplateResponse("fragments/chat.html", {"request": request})


@app.get("/config", response_class=HTMLResponse)
def get_config(request: Request):
    """Configuration page."""
    try:
        with open(CONFIG_PATH, "r") as f:
            config_content = f.read()
    except FileNotFoundError:
        config_content = "# config.toml not found\n"
    return templates.TemplateResponse(
        "fragments/config.html", {"request": request, "config_content": config_content}
    )


@app.get("/data", response_class=HTMLResponse)
def get_data_page(request: Request):
    """List audio and document files."""
    audio_files = sorted(os.listdir(AUDIO_FOLDER))
    doc_files = sorted(os.listdir(DOCS_FOLDER))
    return templates.TemplateResponse(
        "fragments/data.html",
        {"request": request, "audio_files": audio_files, "doc_files": doc_files},
    )


# ======================
# POST / Action Routes
# ======================
@app.post("/ask", response_class=HTMLResponse)
def post_ask(request: Request, query: str = Form(...)):
    """Process user query and return answer + retrieved chunks."""
    raw_answer, retrieved_chunks = rag.query(query)
    html_answer = markdown.markdown(raw_answer, extensions=["fenced_code", "tables"])
    retrieved_chunks_html = markdown_chunks(retrieved_chunks)
    return templates.TemplateResponse(
        "fragments/answer.html",
        {
            "request": request,
            "answer": html_answer,
            "retrieved_chunks": retrieved_chunks_html,
        },
    )


@app.post("/save-config", response_class=HTMLResponse)
def post_save_config(config: str = Form(...)):
    """Save config.toml."""
    with open(CONFIG_PATH, "w") as f:
        f.write(config)
    return HTMLResponse("<p>Config saved ✅</p>")


@app.post("/upload-audio", response_class=HTMLResponse)
async def upload_audio(file: UploadFile = File(...)):
    """Upload file to audio folder."""
    filename = save_file(AUDIO_FOLDER, file)
    return HTMLResponse(f"<p>Uploaded to audio folder: {filename} ✅</p>")


@app.post("/upload-doc", response_class=HTMLResponse)
async def upload_doc(file: UploadFile = File(...)):
    """Upload file to document folder."""
    filename = save_file(DOCS_FOLDER, file)
    return HTMLResponse(f"<p>Uploaded to document folder: {filename} ✅</p>")


@app.post("/refresh-kb", response_class=HTMLResponse)
def refresh_knowledge_base():
    """Rebuild knowledge base."""
    rag.build_knowledge_base()
    return HTMLResponse("<p>Knowledge base updated ✅</p>")


@app.post("/process-data", response_class=HTMLResponse)
def process_data_input():
    """Process data sources."""
    rag.process_data_sources()
    return HTMLResponse("<p>Processed data sources ✅</p>")


# ======================
# App Runner
# ======================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
