import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

train_data_path = 'data/wav/test_random5s'
train_save_path = 'CNN/Tâttaaa'

# Define function to create spectrogram
def create_spectrogram(filename, name, save_path):
    # Load audio file
    clip, sample_rate = librosa.load(filename, sr=None)
    # Create and save spectrogram
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spectrogram = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max))
    filename = name + '.png'
    save_file_path = os.path.join(save_path, filename)
    plt.savefig(save_file_path, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close('all')

# Loop through train data directory to create spectrograms
for root, subdirs, files in os.walk(train_data_path):
    for file in files:
        # Check if file is a wav file
        if file.endswith('.wav'):
            # Create spectrogram and save it
            file_path = os.path.join(root, file)
            file_name = os.path.splitext(file)[0]
            save_path = os.path.join(train_save_path, os.path.relpath(root, train_data_path))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            create_spectrogram(file_path, file_name, save_path)
