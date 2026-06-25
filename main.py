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
from signalwire.rest import Client
from signalwire.voice_response import Connect, Stream, VoiceResponse

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
    """Load restaurant configs from restaurants.json, keyed by phone number.

    order_webhook_api_key is a secret, so it is kept out of restaurants.json. Each
    restaurant instead sets order_webhook_api_key_env to the name of the env var
    holding its own key (multiple restaurants -> multiple keys -> multiple env vars).
    A restaurant with no order_webhook_api_key_env falls back to ORDER_WEBHOOK_API_KEY,
    which is convenient while there is only one restaurant configured.
    """
    restaurants_path = os.path.join(DIR_PATH, "restaurants.json")
    with open(restaurants_path, "r", encoding="utf-8") as file:
        restaurants = json.load(file)
    default_webhook_api_key = os.getenv("ORDER_WEBHOOK_API_KEY")
    for restaurant in restaurants:
        if restaurant.get("order_webhook_api_key"):
            continue
        env_var_name = restaurant.get("order_webhook_api_key_env")
        restaurant["order_webhook_api_key"] = (
            os.getenv(env_var_name, "") if env_var_name else (default_webhook_api_key or "")
        )
    return {r["phone_number"]: r for r in restaurants}


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "models/gemini-3.1-flash-live-preview")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Puck")
SIGNALWIRE_PROJECT_ID = os.getenv("SIGNALWIRE_PROJECT_ID")
SIGNALWIRE_TOKEN = os.getenv("SIGNALWIRE_TOKEN")
SIGNALWIRE_SPACE_URL = os.getenv("SIGNALWIRE_SPACE_URL")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
PORT = int(os.getenv("PORT", 8000))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

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

# Gemini Live bills audio input by the minute, and Twilio forwards a media
# frame every 20ms even during dead air (hold music gaps, line noise,
# customer thinking). Once a stretch of near-silence has lasted longer than
# SILENCE_GATE_HANGOVER_MS we stop forwarding it to Gemini -- real speech is
# always above the RMS threshold and gets sent immediately, so this never
# trims actual words, only the dead air between them.
SILENCE_GATE_RMS_THRESHOLD = int(os.getenv("SILENCE_GATE_RMS_THRESHOLD", "300"))
SILENCE_GATE_HANGOVER_MS = float(os.getenv("SILENCE_GATE_HANGOVER_MS", "400"))

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

if not SIGNALWIRE_PROJECT_ID or not SIGNALWIRE_TOKEN or not SIGNALWIRE_SPACE_URL:
    raise ValueError("Missing SignalWire configuration. Please set it in the .env file.")


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
        client = Client(
            SIGNALWIRE_PROJECT_ID,
            SIGNALWIRE_TOKEN,
            signalwire_space_url=SIGNALWIRE_SPACE_URL,
        )
        call = client.calls.create(
            url=f"{PUBLIC_BASE_URL}/outgoing-call?restaurant_id={restaurant['id']}",
            to=request.to_phone_number,
            from_=restaurant["phone_number"],
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
    stream = Stream(url=f"wss://{request.url.hostname}/media-stream")
    stream.parameter(name="restaurant_id", value=restaurant["id"])
    connect.append(stream)
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

    # Custom parameters on <Stream> (set in /outgoing-call) arrive in the "start"
    # event's customParameters, not the connection URL's query string -- some
    # proxies between Twilio and this server don't reliably forward query strings
    # on the WebSocket upgrade request.
    stream_sid = None
    call_sid = None
    restaurant_id = None
    restaurant = None
    async for message in websocket.iter_text():
        data = json.loads(message)
        if data["event"] == "start":
            stream_sid = data["start"]["streamSid"]
            call_sid = data["start"].get("callSid")
            restaurant_id = data["start"].get("customParameters", {}).get("restaurant_id")
            restaurant = RESTAURANTS_BY_ID.get(restaurant_id)
            break

    if not restaurant:
        print(f"Unknown restaurant_id on media-stream: {restaurant_id}")
        await websocket.close()
        return

    print(f"Incoming stream has started {stream_sid}")

    async with websockets.connect(GEMINI_WS_URL) as gemini_ws:
        await send_setup_message(gemini_ws, restaurant)
        upsample_state = None
        downsample_state = None

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to Gemini Live."""
            nonlocal upsample_state
            silence_ms = 0.0
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data["event"] == "media" and gemini_ws.close_code is None:
                        ulaw_bytes = base64.b64decode(data["media"]["payload"])
                        pcm_8k = audioop.ulaw2lin(ulaw_bytes, 2)

                        frame_ms = (len(pcm_8k) / 2) / TWILIO_SAMPLE_RATE * 1000
                        if audioop.rms(pcm_8k, 2) < SILENCE_GATE_RMS_THRESHOLD:
                            silence_ms += frame_ms
                        else:
                            silence_ms = 0.0
                        if silence_ms > SILENCE_GATE_HANGOVER_MS:
                            continue

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
            except WebSocketDisconnect:
                print("Client disconnected.")
                if gemini_ws.close_code is None:
                    await gemini_ws.close()

        async def send_to_twilio():
            """Receive events from Gemini Live, send audio back to Twilio."""
            nonlocal downsample_state
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


async def fetch_menu(restaurant: dict) -> dict | None:
    """Fetch this restaurant's live menu from RestoPOS via rpc/voice_get_menu.

    Without this, the agent only knows whatever products were typed into the
    static prompt and ends up offering items that don't exist (or missing
    real ones); voice_create_order re-validates against the same table when
    the order is submitted, so this is just for what the agent says out loud.
    """
    api_key = restaurant.get("order_webhook_api_key")
    if not SUPABASE_URL or not SUPABASE_ANON_KEY or not api_key:
        return None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/voice_get_menu",
                json={"p_api_key": api_key},
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "Content-Type": "application/json",
                },
            )
        if resp.status_code >= 400:
            print(f"Error fetching menu for '{restaurant['id']}': HTTP {resp.status_code}")
            return None
        return resp.json()
    except Exception as e:
        print(f"Error fetching menu for '{restaurant['id']}': {e}")
        return None


def format_menu(menu: dict) -> str:
    """Render the RestoPOS menu payload as text to append to the system prompt."""
    products = menu.get("products") or []
    if not products:
        return ""

    category_names = {c["id"]: c["name"] for c in menu.get("categories") or []}
    items_by_category: dict[str, list[str]] = {}
    for product in products:
        category_name = category_names.get(product.get("categoryId"), "Otros")
        line = f"- {product['name']}: {product['price']}"
        if product.get("description"):
            line += f" ({product['description']})"
        items_by_category.setdefault(category_name, []).append(line)

    sections = [
        f"{category_name}:\n" + "\n".join(lines)
        for category_name, lines in items_by_category.items()
    ]
    return (
        "MENU ACTUAL (estos son los unicos productos disponibles; usa exactamente "
        "estos nombres y precios, no inventes otros):\n\n" + "\n\n".join(sections)
    )


async def send_setup_message(gemini_ws, restaurant: dict):
    """Configure the Gemini Live session with this restaurant's prompt and tools."""
    system_message = load_prompt(restaurant.get("system_prompt_file", "system_prompt.txt"))
    voice = restaurant.get("voice", GEMINI_VOICE)

    menu = await fetch_menu(restaurant)
    if menu:
        menu_text = format_menu(menu)
        if menu_text:
            system_message = f"{system_message}\n\n{menu_text}"
        extra_prompt = (menu.get("restaurant") or {}).get("extraPrompt")
        if extra_prompt:
            system_message = f"{system_message}\n\n{extra_prompt}"
    else:
        print(
            f"Could not load live menu for restaurant '{restaurant['id']}'; "
            "falling back to the static prompt only."
        )

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
