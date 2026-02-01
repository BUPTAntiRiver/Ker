# Ker

## What is Ker?

Ker is a non-autonomous AI learning companion. It supports users as a tutor, mentor, and friend while staying strictly user-driven. Ker never sets goals, plans tasks, or takes actions on a user's behalf. It only responds to explicit input and can offer explanations, hints, or optional suggestions.

## Features

Current:

- DeepSeek API backend (OpenAI-compatible chat endpoint)
- Console chat interface (REPL)
- Voice input (always listening, user-driven)
- Text-to-speech output with streaming-style chunking

Planned:

- Desktop application with floating icon
- Full text chat window UI
- Screen context awareness (explicit, opt-in)

## Architecture

Ker is split into clear modules:

- UI: chat UI and floating icon
- Agent logic: user-driven reasoning and response handling
- LLM client: DeepSeek API wrapper
- Voice I/O: audio capture and playback
- Screen context: explicit capture when requested

## Running (Console Mode)

Console mode provides a simple REPL for early development.

1) Set environment variables (optional but recommended):
   - DEEPSEEK_API_KEY
   - DEEPSEEK_BASE_URL (default: <https://api.deepseek.com>)
   - KER_MODEL (default: deepseek-chat)
   - KER_TEMPERATURE (default: 0.4)
   - KER_VOICE_GENDER (default: female)
   - KER_VOICE_NAME (optional, substring match)
   - KER_VOICE_ID (optional, explicit engine voice id)
   - KER_VOICE_RATE (optional, integer)
   - KER_VOICE_VOLUME (optional, 0.0-1.0)

2) Run:

   uv run main.py

If no API key is configured, Ker will return a configuration hint message instead of calling the LLM.

## Tests (uv)

Use uv to manage the environment and run tests:

1) Create the environment and install test dependencies:

   uv venv
   uv pip install -e ".[test]"

2) Run tests:

   uv run pytest

## Privacy & Safety

- Screen context is only used when explicitly provided by the user.
- Voice input uses the system microphone via SpeechRecognition; TTS uses pyttsx3.
