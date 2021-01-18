from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

#Show Plot
show = True

# Directory Data Speech
df = "eng"

# Hanya untuk pengetahuan
data_mfcc   = {'03': [], '04': [], '05' : []}
mean_mfcc   = {'03': [], '04': [], '05' : []}
std_mfcc    = {'03': [], '04': [], '05' : []}
std_range   = {'03': {'min' : 9999, 'max' : -99999},
             '04': {'min' : 9999, 'max' : -99999},
             '05': {'min' : 9999, 'max' : -99999}}
# Digunakan untuk training Model
data_fitur  = {'03': [], '04': [], '05' : []}

c = 0
# Looping untuk setiap file dataset yang ada
for i, filename in enumerate(os.listdir(df)):
    if filename.endswith('.wav'):
        # Filename Spesifikasi
        modality = filename.split('-')[0]
        vocal    = filename.split('-')[1]
        emotion  = filename.split('-')[2]
        emo_inter= filename.split('-')[3]
        state    = filename.split('-')[4]
        repeate  = filename.split('-')[5]
        actor    = filename.split('-')[6].split('.')[0]

        # Memilih dataset dari dataset kaggle untuk 
        if modality == '03' and \
           vocal == '01' and \
           emotion in ['03', '04', '05'] and \
           emo_inter == '01' and \
           state == '02' and \
           repeate == '01':
            c+= 1
            
            #read wav
            (bitrate, signal) = wav.read(df+"/"+filename)

            #get filterbank energi
            fbank_feat = logfbank(signal, bitrate)

            #get mfcc
            mfcc_feat = mfcc(signal, bitrate,
                             nfilt=20,
                             nfft=1200)
            # Rata rata
            x_ = np.mean(mfcc_feat)
            
            # Standar Deviasi
            std_ = np.std(mfcc_feat)
            
            # Rata Rata MFCC
            fitur = np.mean(mfcc_feat, axis=0)

            #Insert Data
            data_mfcc[emotion].append(mfcc_feat)
            mean_mfcc[emotion].append(x_)
            std_mfcc[emotion].append(std_)

            # Insert Data Fitur
            data_fitur[emotion].append(fitur)

            # Penerapan Paper 2
            # Mencari nilai range Standart Deviasi
            if std_ < std_range[emotion]['min']:
                std_range[emotion]['min'] = std_
            if std_ > std_range[emotion]['max']:
                std_range[emotion]['max'] = std_

            # Penerapan Paper 1
            if show and c < 3:
                # Membuat ploting untuk 3 buah grafik 
                fig, (ax1, ax2, ax3) = plt.subplots(3)

                # Mengubah isi dari grafik 1 dengan waveform
                ax1.set_title("Speech waveform %s"%(i+1))
                ax1.set(xlabel="Time (s)", ylabel="Aplitudo")
                ax1.plot(signal/10000)

                # Mengubah isi dari grafik 2 dengan nilai filterbank energi
                ax2.set_title("filterbank energi %s"%(i+1))
                ax2.set(xlabel="Time (s)", ylabel="Channel Index")
                ax2.plot(fbank_feat)

                # Mengubah isi dari grafik 3 dengan nilai mfcc
                ax3.set_title("MFCC %s"%(i+1))
                ax3.set(xlabel="Time (s)", ylabel="Cepstrum Index")
                ax3.plot(mfcc_feat)



print(data_mfcc)
print(mean_mfcc)
print(std_mfcc)
print(std_range)
print(data_fitur)


if show:
    plt.show()


# Convert to array data fitur
fitur = []
target = []
for kelas in data_fitur.keys(): # looping nilai kelas happy(3), sad(4), angry(5)
    for data in data_fitur[kelas]:
        fitur.append(data)
        target.append(kelas)


# export dataset fitur
output_file = open("dataset_speech.csv", 'w')

#Membuat header tabel csv
output_file.write('datano,')
for i in range(13):
    output_file.write('fitur%s,' % (i))
output_file.write('kelas\n')

#Mengisi data tabel csv
for i, f in enumerate(fitur):
    output_file.write('%d,' % (i))
    # Looping sebanyak fitur yang ada
    for ft in f:
        output_file.write('%f,' % (ft))
    output_file.write('%s\n' % (target[i]))

output_file.close()
print("Processing finished.")
