from pydub import AudioSegment 
from pydub.silence import detect_nonsilent
from uuid import uuid4
import os

os.makedirs('output',exist_ok=True)

def remove_silence(filname,format):
    waudio = AudioSegment.from_file(filname, format=format)
    non_silent_ranges=detect_nonsilent(waudio,min_silence_len=1000,silence_thresh=-40)
    non_silent_audio=AudioSegment.empty()
    for start,end in non_silent_ranges:
        non_silent_audio+=waudio[start:end]
    output_path=f"output/{uuid4()}.wav"    
    non_silent_audio.export(output_path,format=format)
    return output_path
