import asyncio
import audioop
import base64
import json
import os
from datetime import datetime, timezone

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel
from twilio.rest import Client
from twilio.twiml.voice_response import Connect, VoiceResponse

load_dotenv()

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def load_prompt(file_name: str) -> str:
    """Load a system prompt file from the prompts/ directory."""
    prompt_path = os.path.join(DIR_PATH, "prompts", file_name)
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Could not find file: {prompt_path}")
        raise


def load_restaurants() -> dict:
    """Load restaurant configs from restaurants.json, keyed by Twilio phone number."""
    restaurants_path = os.path.join(DIR_PATH, "restaurants.json")
    with open(restaurants_path, "r", encoding="utf-8") as file:
        restaurants = json.load(file)
    return {r["twilio_phone_number"]: r for r in restaurants}


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "models/gemini-3.1-flash-live-preview")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Puck")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
PORT = int(os.getenv("PORT", 8000))

RESTAURANTS_BY_PHONE = load_restaurants()
RESTAURANTS_BY_ID = {r["id"]: r for r in RESTAURANTS_BY_PHONE.values()}

GEMINI_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={GEMINI_API_KEY}"
)

# Twilio media streams use 8kHz mu-law; Gemini Live expects 16kHz PCM in
# and returns 24kHz PCM out, so audio is resampled/transcoded both ways.
TWILIO_SAMPLE_RATE = 8000
GEMINI_INPUT_SAMPLE_RATE = 16000
GEMINI_OUTPUT_SAMPLE_RATE = 24000

SUBMIT_ORDER_TOOL = {
    "functionDeclarations": [
        {
            "name": "submit_order",
            "description": (
                "Submit the customer's order once all items, the delivery type, "
                "and their contact details have been confirmed out loud with the "
                "customer. Call this only once per order."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "customer_name": {
                        "type": "STRING",
                        "description": "Customer's full name",
                    },
                    "customer_phone": {
                        "type": "STRING",
                        "description": "Customer's callback phone number",
                    },
                    "items": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "name": {
                                    "type": "STRING",
                                    "description": "Menu item name",
                                },
                                "quantity": {"type": "INTEGER"},
                                "notes": {
                                    "type": "STRING",
                                    "description": "Special requests for this item",
                                },
                            },
                            "required": ["name", "quantity"],
                        },
                    },
                    "delivery_type": {
                        "type": "STRING",
                        "enum": ["pickup", "delivery"],
                    },
                    "delivery_address": {
                        "type": "STRING",
                        "description": "Required when delivery_type is delivery",
                    },
                    "special_instructions": {"type": "STRING"},
                },
                "required": [
                    "customer_name",
                    "customer_phone",
                    "items",
                    "delivery_type",
                ],
            },
        }
    ]
}

app = FastAPI()

if not GEMINI_API_KEY:
    raise ValueError("Missing the Gemini API key. Please set it in the .env file.")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise ValueError("Missing Twilio configuration. Please set it in the .env file.")


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "AI Voice Assistant is running!"}


class CallRequest(BaseModel):
    to_phone_number: str
    restaurant_id: str


@app.post("/make-call")
async def make_call(request: CallRequest):
    """Initiate an outbound call to the specified phone number on behalf of a restaurant."""
    restaurant = RESTAURANTS_BY_ID.get(request.restaurant_id)
    if not restaurant:
        return {
            "error": f"Unknown restaurant_id: {request.restaurant_id}",
            "status": "failed",
        }

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            url=f"{PUBLIC_BASE_URL}/outgoing-call?restaurant_id={restaurant['id']}",
            to=request.to_phone_number,
            from_=restaurant["twilio_phone_number"],
            record=True,
            recording_status_callback=f"{PUBLIC_BASE_URL}/recording-status",
            recording_status_callback_method="POST",
        )
        print(f"Call initiated with SID: {call.sid}")
        return {"call_sid": call.sid, "status": "success"}
    except Exception as e:
        print(f"Error initiating call: {e}")
        return {"error": str(e), "status": "failed"}


@app.api_route("/outgoing-call", methods=["GET", "POST"])
async def handle_outgoing_call(request: Request):
    """Handle outgoing/incoming call webhook and return TwiML response."""
    restaurant_id = request.query_params.get("restaurant_id")
    restaurant = RESTAURANTS_BY_ID.get(restaurant_id) if restaurant_id else None

    if not restaurant and request.method == "POST":
        form_data = await request.form()
        restaurant = RESTAURANTS_BY_PHONE.get(form_data.get("To"))

    response = VoiceResponse()

    if not restaurant:
        response.say("Sorry, this number is not configured. Goodbye.")
        response.hangup()
        return HTMLResponse(content=str(response), media_type="application/xml")

    response.say(f"Hello! You are now connected to {restaurant['name']}'s AI assistant.")
    response.pause(length=1)
    response.say("Please wait while we establish the connection.")

    connect = Connect()
    connect.stream(
        url=f"wss://{request.url.hostname}/media-stream?restaurant_id={restaurant['id']}"
    )
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.api_route("/recording-status", methods=["POST"])
async def handle_recording_status(request: Request):
    """Handle recording status updates from Twilio."""
    form_data = await request.form()
    recording_status = form_data.get("RecordingStatus")
    recording_sid = form_data.get("RecordingSid")
    call_sid = form_data.get("CallSid")
    recording_url = form_data.get("RecordingUrl")
    recording_duration = form_data.get("RecordingDuration")

    print(f"Recording status update:")
    print(f"  Call SID: {call_sid}")
    print(f"  Recording SID: {recording_sid}")
    print(f"  Status: {recording_status}")

    if recording_status == "completed":
        print(f"  Recording URL: {recording_url}")
        print(f"  Duration: {recording_duration} seconds")

    return {"status": "received"}


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and Gemini Live."""
    print("Client connected")
    await websocket.accept()

    restaurant_id = websocket.query_params.get("restaurant_id")
    restaurant = RESTAURANTS_BY_ID.get(restaurant_id)
    if not restaurant:
        print(f"Unknown restaurant_id on media-stream: {restaurant_id}")
        await websocket.close()
        return

    async with websockets.connect(GEMINI_WS_URL) as gemini_ws:
        await send_setup_message(gemini_ws, restaurant)
        stream_sid = None
        call_sid = None
        upsample_state = None
        downsample_state = None

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to Gemini Live."""
            nonlocal stream_sid, call_sid, upsample_state
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data["event"] == "media" and gemini_ws.close_code is None:
                        ulaw_bytes = base64.b64decode(data["media"]["payload"])
                        pcm_8k = audioop.ulaw2lin(ulaw_bytes, 2)
                        pcm_16k, upsample_state = audioop.ratecv(
                            pcm_8k,
                            2,
                            1,
                            TWILIO_SAMPLE_RATE,
                            GEMINI_INPUT_SAMPLE_RATE,
                            upsample_state,
                        )
                        audio_message = {
                            "realtimeInput": {
                                "audio": {
                                    "data": base64.b64encode(pcm_16k).decode("utf-8"),
                                    "mimeType": f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE}",
                                }
                            }
                        }
                        await gemini_ws.send(json.dumps(audio_message))
                    elif data["event"] == "start":
                        stream_sid = data["start"]["streamSid"]
                        call_sid = data["start"].get("callSid")
                        print(f"Incoming stream has started {stream_sid}")
            except WebSocketDisconnect:
                print("Client disconnected.")
                if gemini_ws.close_code is None:
                    await gemini_ws.close()

        async def send_to_twilio():
            """Receive events from Gemini Live, send audio back to Twilio."""
            nonlocal stream_sid, downsample_state
            try:
                async for gemini_message in gemini_ws:
                    response = json.loads(gemini_message)

                    if "setupComplete" in response:
                        print("Gemini Live session ready")
                        continue

                    tool_call = response.get("toolCall")
                    if tool_call:
                        await handle_tool_call(gemini_ws, tool_call, restaurant, call_sid)
                        continue

                    server_content = response.get("serverContent")
                    if not server_content:
                        continue

                    if server_content.get("interrupted"):
                        print("Speech started, interrupting AI response")
                        await websocket.send_json(
                            {"streamSid": stream_sid, "event": "clear"}
                        )

                    model_turn = server_content.get("modelTurn")
                    if model_turn:
                        for part in model_turn.get("parts", []):
                            inline_data = part.get("inlineData")
                            if not inline_data or not inline_data.get("data"):
                                continue
                            try:
                                pcm_24k = base64.b64decode(inline_data["data"])
                                pcm_8k, downsample_state = audioop.ratecv(
                                    pcm_24k,
                                    2,
                                    1,
                                    GEMINI_OUTPUT_SAMPLE_RATE,
                                    TWILIO_SAMPLE_RATE,
                                    downsample_state,
                                )
                                ulaw_bytes = audioop.lin2ulaw(pcm_8k, 2)
                                audio_delta = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": base64.b64encode(
                                            ulaw_bytes
                                        ).decode("utf-8")
                                    },
                                }
                                await websocket.send_json(audio_delta)
                            except Exception as e:
                                print(f"Error processing audio data: {e}")
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        await asyncio.gather(receive_from_twilio(), send_to_twilio())


async def handle_tool_call(gemini_ws, tool_call: dict, restaurant: dict, call_sid):
    """Execute the functions Gemini Live asked for and report the results back."""
    function_responses = []
    for function_call in tool_call.get("functionCalls", []):
        name = function_call.get("name")
        args = function_call.get("args", {})
        call_id = function_call.get("id")

        if name == "submit_order":
            result = await submit_order(restaurant, args, call_sid)
        else:
            result = {"status": "error", "message": f"Unknown function: {name}"}

        function_responses.append({"id": call_id, "name": name, "response": result})

    await gemini_ws.send(
        json.dumps({"toolResponse": {"functionResponses": function_responses}})
    )


async def submit_order(restaurant: dict, order_args: dict, call_sid) -> dict:
    """POST the collected order to the restaurant's own order webhook."""
    webhook_url = restaurant.get("order_webhook_url")
    if not webhook_url:
        return {
            "status": "error",
            "message": "No order webhook configured for this restaurant.",
        }

    payload = {
        "event": "order.created",
        "restaurant_id": restaurant["id"],
        "call_sid": call_sid,
        "customer": {
            "name": order_args.get("customer_name"),
            "phone": order_args.get("customer_phone"),
        },
        "order": {
            "items": order_args.get("items", []),
            "delivery_type": order_args.get("delivery_type"),
            "delivery_address": order_args.get("delivery_address"),
            "special_instructions": order_args.get("special_instructions"),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    headers = {"Content-Type": "application/json"}
    api_key = restaurant.get("order_webhook_api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"Submitting order for restaurant '{restaurant['id']}': {payload}")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(webhook_url, json=payload, headers=headers)
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if "status" in body:
            return body
        if resp.status_code >= 400:
            return {"status": "error", "message": f"HTTP {resp.status_code}"}
        return body or {"status": "ok"}
    except Exception as e:
        print(f"Error submitting order to {webhook_url}: {e}")
        return {"status": "error", "message": str(e)}


async def send_setup_message(gemini_ws, restaurant: dict):
    """Configure the Gemini Live session with this restaurant's prompt and tools."""
    system_message = load_prompt(restaurant.get("system_prompt_file", "system_prompt.txt"))
    voice = restaurant.get("voice", GEMINI_VOICE)

    setup_message = {
        "setup": {
            "model": GEMINI_LIVE_MODEL,
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "temperature": 0.2,
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                },
            },
            "systemInstruction": {"parts": [{"text": system_message}]},
            "tools": [SUBMIT_ORDER_TOOL],
        }
    }
    print(f"Configuring Gemini Live session for restaurant '{restaurant['id']}'")
    await gemini_ws.send(json.dumps(setup_message))
