import os
import requests

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")


def send_message(phone: str, message: str):
    url = f"https://graph.facebook.com/v23.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "text",
        "text": {
            "body": message
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    print(response.status_code)
    print(response.text)


# ------------------------
# Webhook Verification
# ------------------------
@app.get("/webhook")
async def verify(
    hub_mode: str = None,
    hub_verify_token: str = None,
    hub_challenge: str = None,
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge)

    raise HTTPException(status_code=403, detail="Verification Failed")


# ------------------------
# Incoming Messages
# ------------------------
@app.post("/webhook")
async def webhook(request: Request):

    body = await request.json()

    try:
        message = body["entry"][0]["changes"][0]["value"]["messages"][0]

        sender = message["from"]

        print(f"Message received from {sender}")

        send_message(
            sender,
            "Hello Rahul! Welcome to the Transport service of GCS."
        )

    except Exception as e:
        print("Ignored:", e)

    return {"status": "ok"}