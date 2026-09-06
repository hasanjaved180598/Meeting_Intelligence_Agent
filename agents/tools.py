
import whisper


_whisper_model = None

WHISPER_MODEL_SIZE = "base"


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:

    try:
        model = _get_model()
        result = model.transcribe(audio_path)
        text = result.get("text", "").strip()
        if not text:
            raise ValueError("Transcription produced no text — audio may be silent or unreadable.")
        return text
    except Exception as e:
        raise RuntimeError(f"Transcription failed for '{audio_path}': {e}") from e
