import os
import math
from pydub import AudioSegment

def split_audio_file(input_file_path, output_folder_path, split_length_sec):
    # Load input audio file
    sound = AudioSegment.from_file(input_file_path)

    # Calculate total number of split files
    total_split_files = math.ceil(sound.duration_seconds / split_length_sec)

    # Split audio file into smaller parts
    for i in range(total_split_files):
        start_sec = i * split_length_sec * 1000
        end_sec = (i + 1) * split_length_sec * 1000
        if end_sec > len(sound):
            end_sec = len(sound)
        split_file_name = os.path.join(output_folder_path, '{}_{}.wav'.format(os.path.splitext(os.path.basename(input_file_path))[0], i+1))
        split_sound = sound[start_sec:end_sec]
        split_sound.export(split_file_name, format="wav")
        
train_data_path = 'data/wav/train_100cut10s/'
wav_train_data_path = 'data/wav/train_100/'

# Create train folder if it does not exist
if not os.path.exists(train_data_path):
    os.makedirs(train_data_path)

# Loop through files in wav_train_data_path directory and split them into 30 seconds parts
for file_name in os.listdir(wav_train_data_path):
    input_file_path = os.path.join(wav_train_data_path, file_name)
    output_folder_path = os.path.join(train_data_path, os.path.splitext(file_name)[0])
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    split_audio_file(input_file_path, output_folder_path,10)
