# Ker

## What is Ker?

Ker is a non-autonomous AI learning companion. It supports users as a tutor, mentor, and friend while staying strictly user-driven. Ker never sets goals, plans tasks, or takes actions on a user's behalf. It only responds to explicit input and can offer explanations, hints, or optional suggestions.

## Features (Planned)

- DeepSeek API backend
- Desktop application with floating icon
- Voice input (always listening, user-driven)
- Text chat interface
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

2) Run:

   python main.py

If no API key is configured, Ker will return a configuration hint message instead of calling the LLM.

## Privacy & Safety

- Screen context is only used when explicitly provided by the user.
- Voice and screen subsystems are placeholders and do not capture data yet.
