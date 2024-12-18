import os
import wav_reader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

audio_path = r'data/mini_vggvox'

# collecting the wav files path and creating list of path and speaker ID for each
all_wav_path = []
speaker_id_list = []

for i in os.walk(audio_path):
    temp_i_0 = i[0].replace('\\', '/')
    possible_id = temp_i_0.split('/')[-2]
    if possible_id[:2] == 'id':
        all_wav_path += list(np.char.array(i[0])+'/'+i[2])
        speaker_id_list += [possible_id]*len(i[2])

all_wav_path.sort()
speaker_id_list.sort()

wav_path_train, wav_path_test, speaker_id_train, speaker_id_test = train_test_split(all_wav_path, speaker_id_list, test_size=0.2, random_state=4)
wav_path_test, wav_path_val, speaker_id_test, speaker_id_val = train_test_split(wav_path_test, speaker_id_test, test_size=0.5, random_state=4)

print(wav_path_train)
print(wav_path_test)
print(wav_path_val)

id_set = np.unique(speaker_id_list)
main_path = 'pickles\\'

max_file_size = 500
for data_type in ['train', 'test', 'val']:

    dir_path = main_path + data_type + '_files_npy'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    wav_path = eval('wav_path_' + data_type)
    speaker_id = eval('speaker_id_' + data_type)
    slice_num = int(len(wav_path)/max_file_size)+1

    for i in range(slice_num):
        x_batch, y_batch = [], []
        start, end = i * max_file_size, (i + 1) * max_file_size
        batch_files, batch_speaker_id = np.array(wav_path)[start: end], np.array(speaker_id)[start: end]

        for j in tqdm(range(len(batch_files))):
            filename, s_id = batch_files[j], batch_speaker_id[j]
            temp_x = wav_reader.get_fft_spectrum(filename)
            x_batch += temp_x
            y_batch += [s_id] * len(temp_x)

        x_batch, y_batch = np.array(x_batch), np.array(y_batch)
        np.save(dir_path + '/y_mini_'+str(i)+'.npy', y_batch)
        np.save(dir_path + '/x_mini_'+str(i)+'.npy', x_batch)
