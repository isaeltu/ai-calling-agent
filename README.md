# AI Calling Agent

A real-time voice AI system that integrates OpenAI's Realtime API with Twilio Voice to create intelligent voice conversations. Perfect for customer service, compliance monitoring, and automated calling systems.

## Branches

- **[main](https://github.com/intellwe/ai-calling-agent/tree/main)** - OpenAI Realtime API version (streaming, low latency)
- **[llama3](https://github.com/intellwe/ai-calling-agent/tree/llama3)** - Llama3 via Together AI (traditional, cost-effective)

## Features

- **Real-time Voice Processing** - Instant speech recognition and response
- **Smart Interruption Handling** - Natural conversation flow with speech detection
- **Flexible Configuration** - Customizable prompts and voice settings
- **Call Recording** - Automatic recording with compliance features
- **WebSocket Communication** - Low-latency audio streaming
- **Production Ready** - Built with FastAPI for scalability

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (with Realtime API access)
- Twilio account (SID, Auth Token, Phone Number)
- ngrok or similar tunneling tool

### Installation

1. **Clone the repository**

```bash
   git clone https://github.com/intellwe/ai-calling-agent.git
   cd ai-calling-agent
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

4. **Start the server**

   ```bash
   uvicorn main:app --port 8000
   ```

5. **Expose with ngrok**
   ```bash
   ngrok http 8000
   ```

## Configuration

Create a `.env` file with the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
PUBLIC_BASE_URL=your_ngrok_or_deployed_url
PORT=8000
```

## API Endpoints

| Method    | Endpoint         | Description               |
| --------- | ---------------- | ------------------------- |
| GET       | `/`              | Health check              |
| POST      | `/make-call`     | Initiate outbound call    |
| POST      | `/outgoing-call` | Twilio webhook handler    |
| WebSocket | `/media-stream`  | Real-time audio streaming |

### Making a Call

```bash
curl -X POST "http://localhost:8000/make-call" \
  -H "Content-Type: application/json" \
  -d '{"to_phone_number": "+1234567890"}'
```

## Architecture

```
┌─────────────┐    WebSocket   ┌─────────────┐    HTTP/WS    ┌─────────────┐
│   Twilio    │ ◄────────────► │  FastAPI    │ ◄───────────► │   Gemini    │
│   Voice     │                │   Server    │               │  Live API   │
└─────────────┘                └─────────────┘               └─────────────┘
```

The system creates a bridge between Twilio's voice services and Gemini's Live API, enabling natural voice conversations with AI. Twilio streams 8kHz mu-law audio, which the server resamples to/from the 16kHz/24kHz PCM that Gemini Live expects.

## Deployment

This repo includes a `Procfile` for platforms like Railway:

1. Push the repo to GitHub and connect it in Railway (it auto-deploys on every push).
2. In the Railway project's **Variables** tab, set `GEMINI_API_KEY`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`. Railway sets `PORT` automatically.
3. Once deployed, set `PUBLIC_BASE_URL` to the public URL Railway gives the service (e.g. `https://your-app.up.railway.app`) — this replaces ngrok and is what Twilio calls for `/outgoing-call` and `/recording-status`.
4. `ngrok` is then only needed for local development.

## Development

### Setup Development Environment

1. **Install development dependencies**

   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Install pre-commit hooks** (optional)
   ```bash
   pre-commit install
   ```

### Code Quality Tools

- **Format code**: `black .`
- **Sort imports**: `isort .`
- **Lint code**: `flake8`
- **Type checking**: `mypy main.py`
- **Security scan**: `bandit -r .`
- **Run tests**: `pytest`

### Customizing AI Behavior

Edit `prompts/system_prompt.txt` to modify the AI's personality and responses.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- [@FardinHash](https://github.com/FardinHash) -> [LinkedIn](https://linkedin.com/in/fardinkai)

- [@RianaAzad](https://github.com/RianaAzad) -> [LinkedIn](https://linkedin.com/in/riana-azad)

## ⚠️ Disclaimer

This project is not officially affiliated with OpenAI or Twilio. Use responsibly and in accordance with their terms of service.

---

⭐ If you find this project helpful, please give it a star!
