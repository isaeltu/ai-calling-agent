import asyncio
import audioop
import base64
import json
import os

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel
from twilio.rest import Client
from twilio.twiml.voice_response import Connect, VoiceResponse

load_dotenv()


def load_prompt(file_name: str) -> str:
    """Load system prompt from file."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    prompt_path = os.path.join(dir_path, "prompts", f"{file_name}.txt")

    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Could not find file: {prompt_path}")
        raise


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "models/gemini-3.1-flash-live-preview")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Puck")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
PORT = int(os.getenv("PORT", 8000))

SYSTEM_MESSAGE = load_prompt("system_prompt")

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

app = FastAPI()

if not GEMINI_API_KEY:
    raise ValueError("Missing the Gemini API key. Please set it in the .env file.")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
    raise ValueError("Missing Twilio configuration. Please set it in the .env file.")


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "AI Voice Assistant is running!"}


class CallRequest(BaseModel):
    to_phone_number: str


@app.post("/make-call")
async def make_call(request: CallRequest):
    """Initiate an outbound call to the specified phone number."""
    if not request.to_phone_number:
        return {"error": "Phone number is required"}

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            url=f"{PUBLIC_BASE_URL}/outgoing-call",
            to=request.to_phone_number,
            from_=TWILIO_PHONE_NUMBER,
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
    """Handle outgoing call webhook and return TwiML response."""
    response = VoiceResponse()
    response.say("Hello! You are now connected to an AI assistant.")
    response.pause(length=1)
    response.say("Please wait while we establish the connection.")

    connect = Connect()
    connect.stream(url=f"wss://{request.url.hostname}/media-stream")
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

    async with websockets.connect(GEMINI_WS_URL) as gemini_ws:
        await send_setup_message(gemini_ws)
        stream_sid = None
        upsample_state = None
        downsample_state = None

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to Gemini Live."""
            nonlocal stream_sid, upsample_state
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


async def send_setup_message(gemini_ws):
    """Configure the Gemini Live session with audio settings and instructions."""
    setup_message = {
        "setup": {
            "model": GEMINI_LIVE_MODEL,
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "temperature": 0.2,
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": GEMINI_VOICE}
                    }
                },
            },
            "systemInstruction": {"parts": [{"text": SYSTEM_MESSAGE}]},
        }
    }
    print("Configuring Gemini Live session")
    await gemini_ws.send(json.dumps(setup_message))
