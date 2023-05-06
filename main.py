import uvicorn
from fastapi import FastAPI
from chatplm.helpers.load_data import load_data
from chatplm.model import ChatPLM


description = "API for ChatPLM - A simple neural network trained on PLM data for university-specific information queries."

app = FastAPI(
    title="ChatPLM API",
    description=description,
    version="0.0.1"
)
data = load_data()
model = ChatPLM(data)


@app.get("/")
def read_root():
    return {"msg": "This is ChatPLM API, go to /docs for more info."}


@app.get("/chat/{prompt}")
def chat(prompt: str):
    return {"output": model.response_from_model(prompt)[0]}
