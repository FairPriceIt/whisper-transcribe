import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
import os
import requests
from uuid import uuid4

model=whisper.load_model('large')

SAVED_DIRECTORY='downloads'
os.makedirs(SAVED_DIRECTORY,exist_ok=True)

app = Flask(__name__)
cors=CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/detect-lang',methods=['POST'])
@cross_origin()
def detect_lang():
    audio_url=request.form.get('url')
    # download the file
    response = requests.get(audio_url,stream=True)
    response.raise_for_status()
    filename=f"{SAVED_DIRECTORY}/{uuid4()}"
    with open(filename,'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    # whisper.load_audio(filename)        
    audio=whisper.load_audio(filename)
    audio=whisper.pad_or_trim(audio)
    mel=whisper.log_mel_spectrogram(audio,n_mels=model.dims.n_mels).to(model.device)
    _,probe=model.detect_language(mel)
    res={"lang":max(probe,key=probe.get)}
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0',port=80)
