#%%
import src.predict as predict
# %%

MODEL = predict.Predictor()
MODEL.setup()
# %%


model_options = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
transcription_options = ["plain text", "srt", "vtt"]

# gradio app with 3 inputs and 1 output, using the predict function


def app_predict(video_file, model_name, transcription):
    print(f"Inference on {video_file} with model {model_name} and transcription {transcription}")
    whisper_results = MODEL.predict( video_file=video_file, 
                                    model_name=model_name,
                                    transcription=transcription)

    return whisper_results["transcription"]

import gradio as gr

gradio_app = gr.Interface(
    fn=app_predict,
    inputs=[
        gr.Video( label="Video file"),
        gr.Dropdown(choices=model_options, label="Model Type",),
        gr.Dropdown(
            choices=transcription_options,
            label="Transcription Option",
        )
    ],
    outputs=[
        gr.Textbox(label="Transcription")
    ],
    title="Transcription App",
    description="Transcribe a video file using the model.",
    allow_flagging = False,
)
# %%
gradio_app.launch(server_port=8888)
# %%
