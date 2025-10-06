import time
import requests
import urllib
from bs4 import BeautifulSoup
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import joblib
import datetime
from datetime import datetime, timezone
from sign import request
import hashlib
import hmac
from urllib.parse import quote
import requests
base_url = ''
MODEL_PATH = ""
MODEL_PATH2 = ''
model = joblib.load(MODEL_PATH)
model2 = joblib.load(MODEL_PATH2)

AK = ""
SK = ""
# def generate_audio_by_prompt(payload):
#     """Generate audio using the API, with added error handling and logging."""
#     url = f"{base_url}/api/generate"
#     try:
#         print(f"Sending request to: {url}")
#         print(f"Payload: {payload}")
#
#         response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
#         print(f"Response status code: {response.status_code}")
#
#         # Check if the response status code indicates success
#         if response.status_code != 200:
#             print(f"Request failed with status code: {response.status_code}")
#             print(f"Response content: {response.text}")
#             return None
#
#         # Attempt to parse JSON response
#         try:
#             data = response.json()
#             print("JSON response successfully parsed.")
#             return data
#         except requests.exceptions.JSONDecodeError:
#             print("Failed to decode JSON. Response content:")
#             print(response.text)
#             return None
#     except requests.exceptions.RequestException as e:
#         print(f"Request exception occurred: {e}")
#         return None
#
def generate_audio_by_prompt(payload):
    url = f"{base_url}/api/generate"
    headers = {
        'Content-Type': 'application/json',
        'Cookie': ''  # Replace with your actual cookie
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request exception occurred: {e}")
        return None

def fetch_audio_url(song_page_url):
    """Fetch the audio URL from the song page."""
    try:
        print(f"Fetching audio URL from: {song_page_url}")
        response = requests.get(song_page_url)
        response.raise_for_status()

        # Parse the webpage content to find the audio URL
        soup = BeautifulSoup(response.text, 'html.parser')
        audio_tag = soup.find('meta', {'property': 'og:audio'})
        if audio_tag and 'content' in audio_tag.attrs:
            print("Audio URL successfully found.")
            return audio_tag['content']

        raise ValueError("Could not find the audio URL on the page.")
    except Exception as e:
        print(f"Error fetching audio URL: {e}")
        return None


def download_and_convert_audio(audio_url, output_path):
    """Download and convert audio to WAV format."""
    try:
        print(f"Downloading audio from: {audio_url}")
        response = requests.get(audio_url)
        response.raise_for_status()

        # Convert audio to WAV format
        audio_bytes = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_bytes)
        audio.export(output_path, format="wav")
        print(f"Audio has been converted and saved to {output_path}")
    except Exception as e:
        print(f"Error downloading or converting audio: {e}")


def make_volcano_request(action, body, query, method):
    now = datetime.utcnow()
    ak = AK
    sk = SK
    return request(method, now, query, {}, ak, sk, action, body)


def get_audio_information(audio_ids):
    """Fetch audio information using the API."""
    url = f"{base_url}/api/get?ids={audio_ids}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching audio information: {e}")
        return None


def process_audio_features(audio_id):
    """Process audio features using OpenSMILE and analyze with the model."""
    try:
        import opensmile
        import pandas as pd

        input_file = f"/{audio_id}.mp3"
        output_csv = f"/{audio_id}.csv"

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.emobase,
            feature_level=opensmile.FeatureLevel.Functionals
        )
        features = smile.process_file(input_file)
        pd.DataFrame(features).to_csv(output_csv, index=False)
        print(f"Audio features saved to {output_csv}")

        # Predict using the model
        data = pd.read_csv(output_csv)
        valence_features = data.iloc[2:, 0].values.reshape(-1, 1)
        predictions = model.predict(valence_features.T)
        print(f"Predictions: {predictions}")
    except Exception as e:
        print(f"Error processing audio features: {e}")

def analyze_music(model, model2, file_path, id):
    """Process audio features using OpenSMILE and analyze with the model."""
    try:
        import opensmile
        import pandas as pd

        input_file = file_path
        audio_id = id
        output_csv =  audio_id + ".csv"

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.emobase,
            feature_level=opensmile.FeatureLevel.Functionals,
            resample=True,
            sampling_rate=48000
        )
        features = smile.process_file(input_file)
        pd.DataFrame(features).to_csv(output_csv, index=False)
        print(f"Audio features saved to {output_csv}")

        index = np.load("valence_select.npy")
        # Predict using the model
        data = pd.read_csv(output_csv)
        data1 = data.values
        index = index.reshape(1,988)
        new_data = data1[index]
        new_data = new_data.reshape(1,100)
        predictions = model.predict(new_data)

        index2 = np.load("arousal_select.npy")
        index2 = index2.reshape(1,988)
        new_data2 = data1[index2]
        new_data2 = new_data2.reshape(1,100)
        predictions2 = model2.predict(new_data2)

        return [predictions[0][0],predictions2[0][0]]
    except Exception as e:
        print(f"Error processing audio features: {e}")
        return -1

if __name__ == "__main__":
    analyze_music(model, "1.wav","2.wav")
