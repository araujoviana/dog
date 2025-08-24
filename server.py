from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from main import RAGPipeline


app = FastAPI()

rag = RAGPipeline()
rag.build_knowledge_base()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_class=HTMLResponse)
def post_ask(request: Request, query: str = Form(...)):
    answer = rag.query(query)
    return templates.TemplateResponse(
        "fragments/answer.html", {"request": request, "answer": answer}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
