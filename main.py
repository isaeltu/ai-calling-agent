import asyncio
import audioop
import base64
import json
import os
import time
from datetime import datetime, timezone

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel
from twilio.rest import Client
from twilio.twiml.voice_response import Connect, Stream, VoiceResponse

load_dotenv()

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PROMPT_CACHE: dict[str, str] = {}
MENU_CACHE: dict[str, dict] = {}
MENU_REFRESH_TASKS: set[asyncio.Task] = set()
MENU_REFRESHING_RESTAURANTS: set[str] = set()


def load_prompt(file_name: str) -> str:
    """Load a system prompt file from the prompts/ directory."""
    if file_name in PROMPT_CACHE:
        return PROMPT_CACHE[file_name]

    prompt_path = os.path.join(DIR_PATH, "prompts", file_name)
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt = file.read().strip()
            PROMPT_CACHE[file_name] = prompt
            return prompt
    except FileNotFoundError:
        print(f"Could not find file: {prompt_path}")
        raise


def load_restaurants() -> dict:
    """Load restaurant configs from restaurants.json, keyed by Twilio phone number.

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
    return {r["twilio_phone_number"]: r for r in restaurants}


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "models/gemini-3.1-flash-live-preview")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Puck")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
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
SILENCE_GATE_HANGOVER_MS = float(os.getenv("SILENCE_GATE_HANGOVER_MS", "250"))

# Gemini Live's default VAD is tuned for dictation, not a phone call -- it waits
# longer than a caller expects before deciding they're done talking. Lowering
# silenceDurationMs (and raising end-of-speech sensitivity) makes it commit to
# "the customer finished" sooner, so the agent starts answering faster. Lowering
# prefixPaddingMs (start-of-speech sensitivity) shaves the same delay off the
# other end, at the start of each utterance.
GEMINI_END_OF_SPEECH_SILENCE_MS = int(os.getenv("GEMINI_END_OF_SPEECH_SILENCE_MS", "120"))
GEMINI_START_OF_SPEECH_PADDING_MS = int(os.getenv("GEMINI_START_OF_SPEECH_PADDING_MS", "50"))

# Safety net, not the primary brevity control (that's the prompt's "1-2
# sentences" instruction, which works most of the time). This preview model
# occasionally gets stuck generating mostly silence/filler audio for very
# little new spoken text -- doubling this budget in testing made a stuck
# response run *longer* (17s vs 7s) without producing meaningfully more
# words, so a low cap that cuts the rare bad case off quickly beats a high
# one hoping it "finishes". Normal replies finish in ~75-100 tokens (~3-4s),
# well under this.
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "140"))

# Menu lookups are the main non-model dependency in the hot path. Keep them
# short, cache successful responses, and use stale-but-real data while refreshing
# in the background so calls do not wait on Supabase when the menu was just read.
MENU_CACHE_TTL_SECONDS = float(os.getenv("MENU_CACHE_TTL_SECONDS", "300"))
MENU_STALE_TTL_SECONDS = float(os.getenv("MENU_STALE_TTL_SECONDS", "3600"))
MENU_FETCH_TIMEOUT_SECONDS = float(os.getenv("MENU_FETCH_TIMEOUT_SECONDS", "1.0"))
MENU_MAX_ITEMS_IN_PROMPT = int(os.getenv("MENU_MAX_ITEMS_IN_PROMPT", "120"))
ORDER_WEBHOOK_TIMEOUT_SECONDS = float(os.getenv("ORDER_WEBHOOK_TIMEOUT_SECONDS", "3.0"))
CALL_LOG_TIMEOUT_SECONDS = float(os.getenv("CALL_LOG_TIMEOUT_SECONDS", "2.0"))

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


@app.on_event("startup")
async def warm_restaurant_menus() -> None:
    """Preload live menus so the first call does not wait on the database."""
    if not RESTAURANTS_BY_ID:
        return

    results = await asyncio.gather(
        *(refresh_menu_cache(restaurant) for restaurant in RESTAURANTS_BY_ID.values()),
        return_exceptions=True,
    )
    loaded = sum(1 for result in results if isinstance(result, dict))
    print(f"Preloaded {loaded}/{len(RESTAURANTS_BY_ID)} restaurant menu cache(s)")


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

    customer_phone = None
    if request.method == "POST":
        form_data = await request.form()
        if not restaurant:
            restaurant = RESTAURANTS_BY_PHONE.get(form_data.get("To"))
        restaurant_number = restaurant.get("twilio_phone_number") if restaurant else None
        for field in (form_data.get("From"), form_data.get("To")):
            if field and field != restaurant_number:
                customer_phone = field
                break

    response = VoiceResponse()

    if not restaurant:
        response.say("Sorry, this number is not configured. Goodbye.")
        response.hangup()
        return HTMLResponse(content=str(response), media_type="application/xml")

    connect = Connect()
    stream = Stream(url=f"wss://{request.url.hostname}/media-stream")
    stream.parameter(name="restaurant_id", value=restaurant["id"])
    if customer_phone:
        stream.parameter(name="customer_phone", value=customer_phone)
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
    customer_phone = None
    async for message in websocket.iter_text():
        data = json.loads(message)
        if data["event"] == "start":
            stream_sid = data["start"]["streamSid"]
            call_sid = data["start"].get("callSid")
            custom_params = data["start"].get("customParameters", {})
            restaurant_id = custom_params.get("restaurant_id")
            customer_phone = custom_params.get("customer_phone")
            restaurant = RESTAURANTS_BY_ID.get(restaurant_id)
            break

    if not restaurant:
        print(f"Unknown restaurant_id on media-stream: {restaurant_id}")
        await websocket.close()
        return

    print(f"Incoming stream has started {stream_sid}")

    upsample_state = None
    downsample_state = None
    # Wall-clock markers used only to log real turn-around latency (last
    # moment we saw real speech -> first response audio byte) so it can be
    # read straight out of Railway logs instead of guessed at.
    last_speech_end = None
    call_started_at = datetime.now(timezone.utc)
    transcript = []
    pending_role = None
    pending_text = ""
    # The Live API is preview-only right now (no stable/GA native-audio model
    # exists), and preview models are subject to Google-side capacity limits
    # shared across all users, not just our own quota -- this project's quota
    # dashboard can show near-zero usage and we'll still get a mid-call 1011
    # "Resource has been exhausted" close. Reconnecting with the session's
    # resumption handle (instead of just letting the call go silent) is the
    # one thing actually in our control here.
    session_resumption_handle = None
    MAX_GEMINI_RECONNECTS = 2

    def flush_pending_transcript():
        nonlocal pending_role, pending_text
        if pending_role and pending_text.strip():
            transcript.append({"role": pending_role, "text": pending_text.strip()})
        pending_role = None
        pending_text = ""

    for attempt in range(MAX_GEMINI_RECONNECTS + 1):
        is_resume = session_resumption_handle is not None
        try:
            async with websockets.connect(GEMINI_WS_URL) as gemini_ws:
                await send_setup_message(gemini_ws, restaurant, resume_handle=session_resumption_handle)

                async def receive_from_twilio():
                    """Receive audio data from Twilio and send it to Gemini Live."""
                    nonlocal upsample_state, last_speech_end
                    silence_ms = 0.0
                    was_speaking = False
                    try:
                        async for message in websocket.iter_text():
                            data = json.loads(message)
                            if data["event"] == "media" and gemini_ws.close_code is None:
                                ulaw_bytes = base64.b64decode(data["media"]["payload"])
                                pcm_8k = audioop.ulaw2lin(ulaw_bytes, 2)

                                frame_ms = (len(pcm_8k) / 2) / TWILIO_SAMPLE_RATE * 1000
                                is_speech = audioop.rms(pcm_8k, 2) >= SILENCE_GATE_RMS_THRESHOLD
                                if is_speech:
                                    silence_ms = 0.0
                                    was_speaking = True
                                else:
                                    if was_speaking:
                                        last_speech_end = time.monotonic()
                                        was_speaking = False
                                    silence_ms += frame_ms
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
                    nonlocal downsample_state, last_speech_end, pending_role, pending_text, session_resumption_handle
                    async for gemini_message in gemini_ws:
                        response = json.loads(gemini_message)

                        resumption_update = response.get("sessionResumptionUpdate")
                        if resumption_update:
                            if resumption_update.get("resumable") and resumption_update.get("newHandle"):
                                session_resumption_handle = resumption_update["newHandle"]
                            continue

                        if "setupComplete" in response:
                            print(f"Gemini Live session ready{' (resumed)' if is_resume else ''}")
                            if not is_resume:
                                await gemini_ws.send(
                                    json.dumps(
                                        {
                                            "clientContent": {
                                                "turns": [
                                                    {
                                                        "role": "user",
                                                        "parts": [
                                                            {
                                                                "text": (
                                                                    "[Inicio de llamada. Saluda al "
                                                                    "cliente ahora mismo: di el nombre "
                                                                    "del restaurante y pregunta en que "
                                                                    "le puedes ayudar hoy. No esperes a "
                                                                    "que el cliente hable primero.]"
                                                                )
                                                            }
                                                        ],
                                                    }
                                                ],
                                                "turnComplete": True,
                                            }
                                        }
                                    )
                                )
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

                        input_transcription = server_content.get("inputTranscription")
                        if input_transcription and input_transcription.get("text"):
                            if pending_role != "customer":
                                flush_pending_transcript()
                                pending_role = "customer"
                            pending_text += input_transcription["text"]

                        output_transcription = server_content.get("outputTranscription")
                        if output_transcription and output_transcription.get("text"):
                            if pending_role != "assistant":
                                flush_pending_transcript()
                                pending_role = "assistant"
                            pending_text += output_transcription["text"]

                        model_turn = server_content.get("modelTurn")
                        if model_turn:
                            for part in model_turn.get("parts", []):
                                inline_data = part.get("inlineData")
                                if not inline_data or not inline_data.get("data"):
                                    continue
                                if last_speech_end is not None:
                                    latency_ms = (time.monotonic() - last_speech_end) * 1000
                                    print(f"Response latency (silence -> first audio byte): {latency_ms:.0f}ms")
                                    last_speech_end = None
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

                tasks = [asyncio.create_task(receive_from_twilio()), asyncio.create_task(send_to_twilio())]
                try:
                    await asyncio.gather(*tasks)
                except Exception:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
            break
        except websockets.exceptions.ConnectionClosedError as e:
            if attempt < MAX_GEMINI_RECONNECTS:
                print(f"Gemini Live connection lost ({e}); reconnecting (attempt {attempt + 2})...")
                await asyncio.sleep(0.5)
                continue
            print(f"Gemini Live connection lost permanently after {attempt + 1} attempts: {e}")
        except Exception as e:
            print(f"Unexpected error in media stream handling: {e}")
            break

    flush_pending_transcript()
    await log_call(restaurant, call_sid, customer_phone, transcript, call_started_at)


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

    validation_error = validate_order_items(restaurant, order_args.get("items", []))
    if validation_error:
        return validation_error

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
        async with httpx.AsyncClient(timeout=ORDER_WEBHOOK_TIMEOUT_SECONDS) as client:
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


async def log_call(restaurant: dict, call_sid, customer_phone, transcript: list, started_at) -> None:
    """Save the call's transcript to RestoPOS via rpc/voice_log_call.

    Best-effort and never raises: today a call that doesn't end in an order
    leaves no trace anywhere, and a logging failure shouldn't be allowed to
    look like a call failure to whoever's debugging this.
    """
    api_key = restaurant.get("order_webhook_api_key")
    if not SUPABASE_URL or not SUPABASE_ANON_KEY or not api_key or not call_sid:
        return

    try:
        async with httpx.AsyncClient(timeout=CALL_LOG_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/voice_log_call",
                json={
                    "p_api_key": api_key,
                    "p_call_sid": call_sid,
                    "p_customer_phone": customer_phone,
                    "p_transcript": transcript,
                    "p_started_at": started_at.isoformat(),
                    "p_ended_at": datetime.now(timezone.utc).isoformat(),
                },
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "Content-Type": "application/json",
                },
            )
        if resp.status_code >= 400:
            print(f"Error logging call '{call_sid}': HTTP {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Error logging call '{call_sid}': {e}")


def build_http_timeout(seconds: float) -> httpx.Timeout:
    """Create a low-latency timeout with a short connection budget."""
    connect_timeout = min(seconds, 0.4)
    pool_timeout = min(seconds, 0.4)
    return httpx.Timeout(
        timeout=seconds,
        connect=connect_timeout,
        read=seconds,
        write=seconds,
        pool=pool_timeout,
    )


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
        async with httpx.AsyncClient(
            timeout=build_http_timeout(MENU_FETCH_TIMEOUT_SECONDS)
        ) as client:
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


def cache_menu(restaurant: dict, menu: dict) -> dict:
    """Store derived menu data so setup and order validation are instant."""
    restaurant_id = restaurant["id"]
    products = menu.get("products") or []
    product_names = {
        str(product.get("name", "")).strip().casefold()
        for product in products
        if product.get("name")
    }
    MENU_CACHE[restaurant_id] = {
        "menu": menu,
        "menu_text": format_menu(menu),
        "product_names": product_names,
        "fetched_at": time.monotonic(),
    }
    return MENU_CACHE[restaurant_id]


async def refresh_menu_cache(restaurant: dict) -> dict | None:
    """Refresh the menu cache from RestoPOS."""
    menu = await fetch_menu(restaurant)
    if not menu:
        return None
    entry = cache_menu(restaurant, menu)
    print(
        f"Menu cache refreshed for '{restaurant['id']}' "
        f"({len(entry['product_names'])} products)"
    )
    return entry


def schedule_menu_refresh(restaurant: dict) -> None:
    """Start a background menu refresh without leaking completed tasks."""
    restaurant_id = restaurant["id"]
    if restaurant_id in MENU_REFRESHING_RESTAURANTS:
        return

    MENU_REFRESHING_RESTAURANTS.add(restaurant_id)
    task = asyncio.create_task(refresh_menu_cache(restaurant))
    MENU_REFRESH_TASKS.add(task)

    def cleanup(done_task: asyncio.Task) -> None:
        MENU_REFRESH_TASKS.discard(done_task)
        MENU_REFRESHING_RESTAURANTS.discard(restaurant_id)
        try:
            done_task.result()
        except Exception as exc:
            print(f"Background menu refresh failed for '{restaurant_id}': {exc}")

    task.add_done_callback(cleanup)


async def get_cached_menu(restaurant: dict) -> dict | None:
    """Return fresh menu data fast, falling back to stale real data if needed."""
    restaurant_id = restaurant["id"]
    entry = MENU_CACHE.get(restaurant_id)
    now = time.monotonic()

    if entry:
        age = now - entry["fetched_at"]
        if age <= MENU_CACHE_TTL_SECONDS:
            return entry
        if age <= MENU_STALE_TTL_SECONDS:
            schedule_menu_refresh(restaurant)
            return entry

    entry = await refresh_menu_cache(restaurant)
    if entry:
        return entry

    stale_entry = MENU_CACHE.get(restaurant_id)
    if stale_entry and now - stale_entry["fetched_at"] <= MENU_STALE_TTL_SECONDS:
        return stale_entry
    return None


def validate_order_items(restaurant: dict, items: list) -> dict | None:
    """Reject model-submitted items that are not in the cached live menu."""
    entry = MENU_CACHE.get(restaurant["id"])
    product_names = entry.get("product_names") if entry else None
    if not product_names:
        return None

    invalid_items = [
        item.get("name")
        for item in items
        if str(item.get("name", "")).strip().casefold() not in product_names
    ]
    if not invalid_items:
        return None

    valid_examples = sorted(entry["product_names"])[:6]
    return {
        "status": "error",
        "message": (
            "No envies el pedido todavia. Estos productos no existen en el menu "
            f"actual: {', '.join(invalid_items)}. Pide al cliente elegir un "
            f"producto real del menu. Ejemplos validos: {', '.join(valid_examples)}."
        ),
    }


def format_menu(menu: dict) -> str:
    """Render the RestoPOS menu payload as text to append to the system prompt."""
    products = menu.get("products") or []
    if not products:
        return ""

    category_names = {c["id"]: c["name"] for c in menu.get("categories") or []}
    items_by_category: dict[str, list[str]] = {}
    for product in products[:MENU_MAX_ITEMS_IN_PROMPT]:
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


async def send_setup_message(gemini_ws, restaurant: dict, resume_handle: str | None = None):
    """Configure the Gemini Live session with this restaurant's prompt and tools."""
    system_message = load_prompt(restaurant.get("system_prompt_file", "system_prompt.txt"))
    system_message = f"Nombre del restaurante: {restaurant['name']}\n\n{system_message}"
    voice = restaurant.get("voice", GEMINI_VOICE)

    menu_entry = await get_cached_menu(restaurant)
    if menu_entry:
        menu_text = menu_entry.get("menu_text")
        if menu_text:
            system_message = f"{system_message}\n\n{menu_text}"
        extra_prompt = (menu_entry["menu"].get("restaurant") or {}).get("extraPrompt")
        if extra_prompt:
            system_message = f"{system_message}\n\n{extra_prompt}"
    else:
        print(
            f"Could not load live menu for restaurant '{restaurant['id']}'; "
            "using static prompt without product names."
        )

    setup_message = {
        "setup": {
            "model": GEMINI_LIVE_MODEL,
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "temperature": 0.2,
                "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                },
            },
            "realtimeInputConfig": {
                "automaticActivityDetection": {
                    "startOfSpeechSensitivity": "START_SENSITIVITY_HIGH",
                    "endOfSpeechSensitivity": "END_SENSITIVITY_HIGH",
                    "prefixPaddingMs": GEMINI_START_OF_SPEECH_PADDING_MS,
                    "silenceDurationMs": GEMINI_END_OF_SPEECH_SILENCE_MS,
                }
            },
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
            "systemInstruction": {"parts": [{"text": system_message}]},
            "tools": [SUBMIT_ORDER_TOOL],
        }
    }
    if resume_handle:
        setup_message["setup"]["sessionResumption"] = {"handle": resume_handle}
    print(
        f"Configuring Gemini Live session for restaurant '{restaurant['id']}'"
        + (" (resuming previous session)" if resume_handle else "")
    )
    await gemini_ws.send(json.dumps(setup_message))
