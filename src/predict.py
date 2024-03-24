"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

# %%
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import os
from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import LANGUAGES
from whisper.utils import format_timestamp

# %%
import ffmpeg


def extract_audio_from_video(outvideo):
    """
    Extract audio from a video file and save it as an MP3 file.

    :param output_video_file: Path to the video file.
    :return: Path to the generated audio file.
    """

    # if video has valid file extension
    file_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]
    if not any([outvideo.endswith(ext) for ext in file_extensions]):
        print("Invalid file extension")
        return None

    # find out video extension
    extension = outvideo.split(".")[-1]

    # replace the file extension with mp3

    audiofilename = outvideo.replace("." + extension, ".mp3")

    # Create the ffmpeg input stream
    input_stream = ffmpeg.input(outvideo)

    # Extract the audio stream from the input stream
    audio = input_stream.audio

    # Save the audio stream as an MP3 file
    output_stream = ffmpeg.output(audio, audiofilename)

    # Overwrite output file if it already exists
    output_stream = ffmpeg.overwrite_output(output_stream)

    ffmpeg.run(output_stream)

    return audiofilename


# %%


class Predictor:
    """A Predictor class for the Whisper model"""

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.models = {}

        def load_model(model_name):
            """
            Load the model from the weights folder.
            """
            try:
                wights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
                model_file = os.path.join(wights_dir, f"{model_name}.pt")
                with open(model_file, "rb") as model_file:
                    checkpoint = torch.load(model_file, map_location="cpu")
                    dims = ModelDimensions(**checkpoint["dims"])
                    model = Whisper(dims)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    return model_name, model
            except FileNotFoundError:
                print(f"Model {model_name} could not be found.")
                return None, None

        model_names = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
        with ThreadPoolExecutor() as executor:
            for model_name, model in executor.map(load_model, model_names):
                if model_name is not None:
                    self.models[model_name] = model

        print("Models loaded")

    def predict(
        self,
        video_file,
        model_name="base",
        transcription="plain text",
        translate=False,
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=None,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps =True
    ):
        audio = extract_audio_from_video(video_file)
        """Run a single prediction on the model"""
        print(f"Transcribe with {model_name} model")
        model = self.models[model_name]
        if torch.cuda.is_available():
            model = model.to("cuda")

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        args = {
            "language": language,
            "best_of": best_of,
            "beam_size": beam_size,
            "patience": patience,
            "length_penalty": length_penalty,
            "suppress_tokens": suppress_tokens,
            "initial_prompt": initial_prompt,
            "condition_on_previous_text": condition_on_previous_text,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "word_timestamps": word_timestamps,
        }

        result = model.transcribe(str(audio), temperature=temperature, **args)

        if transcription == "plain text":
            transcription = result["text"]
        elif transcription == "srt":
            transcription = write_srt(result["segments"])
        else:
            transcription = write_vtt(result["segments"])

        if translate:
            translation = model.transcribe(
                str(audio), task="translate", temperature=temperature, **args
            )

        return {
            "segments": result["segments"],
            "detected_language": LANGUAGES[result["language"]],
            "transcription": transcription,
            "translation": translation["text"] if translate else None,
        }


def write_vtt(transcript):
    """
    Write the transcript in VTT format.
    """
    result = ""
    for segment in transcript:
        result += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result


def write_srt(transcript):
    """
    Write the transcript in SRT format.
    """
    result = ""
    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result

#%%
def extract_segments_from_prediction(prediction):

    wordlevel_info = []
    for item in prediction['segments']:
        words = item.get('words', [])
        for word_info in words:
            wordlevel_info.append({
                'word': word_info.get('word', ''),
                'start': word_info.get('start', 0.0),
                'end': word_info.get('end', 0.0)
            })

    transcript = prediction['transcription']
    return wordlevel_info, transcript


# %%

if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    assets_folder = os.path.join(os.path.dirname(__file__), "..", "assets")
    audio = assets_folder + "/file.mp4"
    prediction = predictor.predict(audio, model_name="base",
                                   word_timestamps=True,
                                   transcription="srt")
    print(prediction)
# %%
