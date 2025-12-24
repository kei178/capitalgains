# Project: Business-Economy Manzai Podcast Automator

## Build and Run Commands
- Install dependencies: `pip install -r requirements.txt`
- Run local test: `python main.py --test`
- Run full pipeline: `python main.py`
- Audio processing check: `ffprobe output.mp3`

## Code Style & Conventions
- **Language**: Python 3.10+
- **Style**: PEP 8 compliant. Use clear, descriptive variable names.
- **Error Handling**: Implement robust try-except blocks, especially for API calls and Audio processing.
- **Logging**: Use the `logging` module to track the flow (RSS fetch -> Selection -> Script -> TTS -> Merge).
- **Docstrings**: Google style docstrings for all functions.

## Project Specific Rules
- **Source of Truth**: Always consult `PROJECT_DESIGN.md` for architecture, data flow, and character personas.
- **Character Voice**: Adhere strictly to the "Capital Gains" persona (Shindo: Onyx/Low-tone, Moriya: Alloy/Fast-tone) as defined in the design doc.
- **Dialect**: Ensure all script text is in natural Kansai dialect.
  