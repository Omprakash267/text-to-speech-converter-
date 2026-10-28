# Tamil Text-to-Speech with IndicF5

This project generates Tamil speech from text using the
[AI4Bharat IndicF5](https://huggingface.co/ai4bharat/IndicF5) model. It uses
`reference.wav` as a voice reference and saves generated audio as WAV files in
`generated_audio`.

## Requirements

- Windows
- Python 3.10 or later
- A Hugging Face account with access to the IndicF5 model and a Hugging Face
  access token
- FFmpeg and FFprobe installed in the WinGet package location expected by
  `speak.py`:
  `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build-shared\bin`
- Python packages: `numpy`, `soundfile`, `transformers`, and the PyTorch
  dependencies required by IndicF5

The script checks for `ffmpeg.exe` and `ffprobe.exe` in that exact directory.
If your FFmpeg installation uses a different location, update the path in
`configure_ffmpeg()` in `speak.py`.

## Setup

1. Open a terminal in the project directory.
2. Create and activate a virtual environment (optional but recommended):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the Python packages:

   ```powershell
   python -m pip install numpy soundfile transformers torch
   ```

   Install any additional packages required by the IndicF5 model if prompted.
   The first model load downloads model files and may take several minutes.

4. Log in to Hugging Face so `token=True` in the script can use your saved
   credentials:

   ```powershell
   hf auth login
   ```

5. Put a non-empty Tamil speech recording at `reference.wav` in the project
   directory. The transcript in `REFERENCE_TEXT` in `speak.py` must accurately
   match the spoken content of that recording.

## Run

```powershell
python speak.py
```

Enter Tamil text when prompted. The program writes each result to
`generated_audio\tamil_speech_NNN.wav`. Type `exit` to stop. Generated audio
uses a 24 kHz sample rate.

## Configuration

The model ID, reference audio path, reference transcript, sample rate, and
output directory are defined near the top of `speak.py`. The reference
transcript should be updated whenever you replace `reference.wav`.
