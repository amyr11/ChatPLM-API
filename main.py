import os
from fastapi import FastAPI, Header, HTTPException
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


API_KEY = os.environ.get("API_KEY")


@app.get("/")
def read_root():
    return {"msg": "This is ChatPLM API, go to /docs for more info."}


@app.get("/chat/")
def chat(prompt: str, api_key: str = Header(None)):
    authorize(api_key)
    response, confidence = model.response_from_model(prompt)
    return {"output": response, "confidence": float(confidence)}


@app.post("/chat/feedback")
async def submit_feedback(
    input_message: str,
    bot_response: str,
    confidence_level: float,
    api_key: str = Header(None)
):
    authorize(api_key)

    save_feedback_to_database(
        input_message, bot_response, confidence_level)

    return {"input_message": input_message, "bot_response": bot_response, "confidence_level": confidence_level}


@app.get("/metadata")
def metadata(api_key: str = Header(None)):
    authorize(api_key)
    return {"trn_updated": data['date']}


def authorize(api_key):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def save_feedback_to_database(input_message, bot_response, confidence_level):
    pass
