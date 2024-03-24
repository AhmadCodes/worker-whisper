''' infer.py for runpod worker '''
#%%
import os
import predict

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup

from dotenv import load_dotenv
from rp_schema import INPUT_VALIDATIONS
import boto3
MODEL = predict.Predictor()
MODEL.setup()
ASSET_DL_PATH = os.path.join(os.path.dirname(__file__), "..", 'assets', "file.mp4")
if os.path.exists('.env'):
    load_dotenv()


def get_s3_client():
    '''
    Get the s3 client.
    '''
    s3_client = boto3.client('s3',
                             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                             aws_secret_access_key=os.getenv(
                                 'AWS_SECRET_ACCESS_KEY'),
                             )
    return s3_client

def download_file_from_s3(bucket, key, s3_client,  local_path = ASSET_DL_PATH):
    '''
    Download the file from s3.
    '''
    local_path_dir = os.path.dirname(local_path)
    if not os.path.exists(local_path_dir):
        os.makedirs(local_path_dir, exist_ok=True)
    s3_client.download_file(bucket, key, local_path)
    
    return local_path

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


#%%

def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']
    s3_client = get_s3_client()

    # Setting the float parameters
    job_input['temperature'] = float(job_input.get('temperature', 0))
    job_input['patience'] = float(job_input.get('patience', 0))
    job_input['length_penalty'] = float(job_input.get('length_penalty', 0))
    job_input['temperature_increment_on_fallback'] = float(
        job_input.get('temperature_increment_on_fallback', 0.2)
    )
    job_input['compression_ratio_threshold'] = float(
        job_input.get('compression_ratio_threshold', 2.4)
    )
    job_input['logprob_threshold'] = float(
        job_input.get('logprob_threshold', -1.0))
    job_input['no_speech_threshold'] = 0.6

    # Input validation
    validated_input = validate(job_input, INPUT_VALIDATIONS)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}

    job_input['video'] = download_file_from_s3(
        job_input['bucket'], job_input['key'], s3_client)
    

    whisper_results = MODEL.predict(
        video_file=job_input["video"],
        model_name=job_input.get("model", 'base'),
        transcription=job_input.get('transcription', 'srt'),
        translate=job_input.get('translate', False),
        language=job_input.get('language', None),
        temperature=job_input["temperature"],
        best_of=job_input.get("best_of", 5),
        beam_size=job_input.get("beam_size", 5),
        patience=job_input["patience"],
        length_penalty=job_input["length_penalty"],
        suppress_tokens=job_input.get("suppress_tokens", "-1"),
        initial_prompt=job_input.get('initial_prompt', None),
        condition_on_previous_text=job_input.get(
            'condition_on_previous_text', True),
        temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
        compression_ratio_threshold=job_input["compression_ratio_threshold"],
        logprob_threshold=job_input["logprob_threshold"],
        no_speech_threshold=job_input["no_speech_threshold"],
        word_timestamps=True
    )

    rp_cleanup.clean(['input_objects'])
    
    word_level_info, transcript = extract_segments_from_prediction(whisper_results)

    returns = {
        "transcript": transcript,
        "word_level_transcript": word_level_info
    }
    
    return returns


#%%
# test_job = {
#     "input": {
#         "bucket": "brollvideo",
#         "key": "y2mate.bz - How Many Earths Could Fit Inside the Sun_.mp4",
#         "model": "base",
#         "transcription": "srt",

#     }
# }

# res = run(test_job)

#%%

runpod.serverless.start({"handler": run})

