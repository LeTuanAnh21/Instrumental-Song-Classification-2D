import os
import random
from pydub import AudioSegment

def random_cut_audio(input_file_path, output_base_folder, duration=10):
    # Load the input audio file
    audio = AudioSegment.from_file(input_file_path)

    # Get the total duration of the audio in milliseconds
    total_duration = len(audio)

    # Calculate the start time for the random cut
    start_time = random.randint(0, total_duration - duration * 1000)

    # Cut the audio from the random start time to the end of the segment
    cut_audio = audio[start_time:start_time + duration * 1000]

    # Get the input file name without extension
    input_file_name = os.path.splitext(os.path.basename(input_file_path))[0]

    # Create the output directory for the current audio
    output_folder_path = os.path.join(output_base_folder, input_file_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Save the cut audio to the output folder
    output_file_path = os.path.join(output_folder_path, f'{input_file_name}.wav')
    cut_audio.export(output_file_path, format='wav')

# Example usage:
input_folder_path = 'data/wav/train_100/'
output_base_folder = 'data/wav/test_train10s/'

for file_name in os.listdir(input_folder_path):
    input_file_path = os.path.join(input_folder_path, file_name)
    random_cut_audio(input_file_path, output_base_folder)
