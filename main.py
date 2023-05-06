import uvicorn
from fastapi import FastAPI
from chatplm.helpers.load_data import load_data
from chatplm.model import ChatPLM

app = FastAPI()
data = load_data()
model = ChatPLM(data)


@app.get("/")
def read_root():
    return {"msg": "This is ChatPLM API, go to /docs for more info."}


@app.get("/chat/{prompt}")
def chat(prompt: str):
    return {"output": model.response_from_model(prompt)[0]}
