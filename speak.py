import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
from transformers import AutoModel


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_ID = "ai4bharat/IndicF5"

BASE_DIR = Path(__file__).resolve().parent

REFERENCE_AUDIO = BASE_DIR / "reference.wav"

OUTPUT_DIR = BASE_DIR / "generated_audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 24000

# This must accurately match the speech in reference.wav.
REFERENCE_TEXT = (
    "வணக்கம்! இன்று ஒரு நல்ல கதையைக் கேட்போம். "
    "தெளிவான சொற்களும் சரியான இடைவெளிகளும் "
    "கதையை இன்னும் இனிமையாக மாற்றும்."
)


# ==========================================
# CONFIGURE FFMPEG AND FFPROBE
# ==========================================

def configure_ffmpeg():
    """Make FFmpeg and FFprobe available to this Python process."""

    local_app_data = os.environ.get("LOCALAPPDATA")

    if not local_app_data:
        raise RuntimeError("Windows LOCALAPPDATA environment variable is missing.")

    ffmpeg_bin = (
        Path(local_app_data)
        / "Microsoft"
        / "WinGet"
        / "Packages"
        / "Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
        / "ffmpeg-9.0.2-full_build-shared"
        / "bin"
    )

    ffmpeg_path = ffmpeg_bin / "ffmpeg.exe"
    ffprobe_path = ffmpeg_bin / "ffprobe.exe"

    if not ffmpeg_path.is_file():
        raise FileNotFoundError(f"FFmpeg not found: {ffmpeg_path}")

    if not ffprobe_path.is_file():
        raise FileNotFoundError(f"FFprobe not found: {ffprobe_path}")

    os.environ["PATH"] = (
        str(ffmpeg_bin) + os.pathsep + os.environ.get("PATH", "")
    )

    print("FFmpeg:", shutil.which("ffmpeg"))
    print("FFprobe:", shutil.which("ffprobe"))


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():
    print("\nLoading IndicF5 model...")
    print("The first load may take several minutes.")
    print("Please wait; model files may need to download.")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=True
    )

    print("Model loaded successfully!")

    return model


# ==========================================
# GENERATE SPEECH
# ==========================================

def generate_speech(model, text, count):
    print("\nGenerating Tamil speech...")

    # IndicF5 uses a reference recording and its transcript.
    audio = model(
        text,
        ref_audio_path=str(REFERENCE_AUDIO),
        ref_text=REFERENCE_TEXT
    )

    audio = np.asarray(audio).squeeze()

    if audio.size == 0:
        raise RuntimeError("The model returned empty audio.")

    if not np.isfinite(audio).all():
        raise RuntimeError("The generated audio contains invalid values.")

    # Convert integer audio to normalized floating-point audio.
    if np.issubdtype(audio.dtype, np.integer):
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            raise RuntimeError(
                f"Unsupported integer audio format: {audio.dtype}"
            )
    else:
        audio = audio.astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)

    output_file = OUTPUT_DIR / f"tamil_speech_{count:03d}.wav"

    sf.write(
        str(output_file),
        audio,
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    duration = len(audio) / SAMPLE_RATE

    print("\nSpeech generated successfully!")
    print(f"Output file: {output_file}")
    print(f"Duration: {duration:.2f} seconds")


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():
    print("=" * 50)
    print("       TAMIL TEXT-TO-SPEECH USING INDICF5")
    print("=" * 50)

    # Configure external audio utilities.
    try:
        configure_ffmpeg()
    except Exception as error:
        print(f"\nFFmpeg configuration failed: {error}")
        return

    # Check reference audio.
    if not REFERENCE_AUDIO.is_file():
        print("\nERROR: reference.wav was not found.")
        print(f"Expected location: {REFERENCE_AUDIO}")
        return

    if REFERENCE_AUDIO.stat().st_size == 0:
        print("\nERROR: reference.wav is empty.")
        return

    # Load the model once.
    try:
        model = load_model()
    except Exception:
        print("\nERROR: Could not load IndicF5.")
        traceback.print_exc()
        return

    print("\nReady to generate Tamil speech.")
    print("Type 'exit' to stop.")

    count = 1

    while True:
        try:
            text = input("\nEnter Tamil text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nProgram stopped.")
            break

        if text.lower() == "exit":
            print("Program stopped.")
            break

        if not text:
            print("Please enter some Tamil text.")
            continue

        try:
            generate_speech(model, text, count)
            count += 1

        except Exception:
            print("\nSpeech generation failed.")
            print("Full error details:")
            traceback.print_exc()


if __name__ == "__main__":
    main()

