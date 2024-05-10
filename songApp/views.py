from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
from scipy.stats import skew,kurtosis
import librosa
from pychorus import find_and_output_chorus
from django.core.files.storage import FileSystemStorage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import load
import warnings
warnings.filterwarnings('ignore')
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import youtube_dl
from django.views.decorators.csrf import csrf_protect




input_file = "./music/"
output_file = "./Choruses/"



def extract_chorus(name):
    if name not in os.listdir(output_file):
        chorus = find_and_output_chorus(f"{input_file}"+f"{name}", f"{output_file}"+f"{name}")

def extract_features(name):
    features = {}
    for feature in ["chroma_stft","chroma_cqt","chroma_cens","mfcc","rms","spectral_centroid","spectral_bandwidth","spectral_contrast","spectral_rolloff","tonnetz","zero_crossing_rate"]:
        if feature == "chroma_stft":
            for i in range(84):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "chroma_cqt":
            for i in range(84):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "chroma_cens":
            for i in range(84):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "mfcc":
            for i in range(140):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "rms":
            for i in range(7):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "spectral_centroid":
            for i in range(7):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "spectral_bandwidth":
            for i in range(7):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "spectral_contrast":
            for i in range(49):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "spectral_rolloff":
            for i in range(7):
                features[f"{feature}_{i+1}"] = []

        if feature == "tonnetz":
            for i in range(37):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "zero_crossing_rate":
            for i in range(7):
                features[f"{feature}_{i+1}"] = []

    y, sr = librosa.load(f"{output_file}"+f"{name}")

    # Extract major audio features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

    for feature in ["chroma_stft","chroma_cqt","chroma_cens","mfcc","rms","spectral_centroid","spectral_bandwidth","spectral_contrast","spectral_rolloff","tonnetz","zero_crossing_rate"]:
        if feature == "chroma_stft":
            min_val = np.min(chroma_stft, axis=1)
            mean_val = np.mean(chroma_stft, axis=1)
            median_val = np.median(chroma_stft, axis=1)
            max_val = np.max(chroma_stft, axis=1)
            std_val = np.std(chroma_stft, axis=1)
            skew_val = skew(chroma_stft, axis=1)
            kurtosis_val = kurtosis(chroma_stft, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "chroma_cqt":
            min_val = np.min(chroma_cqt, axis=1)
            mean_val = np.mean(chroma_cqt, axis=1)
            median_val = np.median(chroma_cqt, axis=1)
            max_val = np.max(chroma_cqt, axis=1)
            std_val = np.std(chroma_cqt, axis=1)
            skew_val = skew(chroma_cqt, axis=1)
            kurtosis_val = kurtosis(chroma_cqt, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "chroma_cens":
            min_val = np.min(chroma_cens, axis=1)
            mean_val = np.mean(chroma_cens, axis=1)
            median_val = np.median(chroma_cens, axis=1)
            max_val = np.max(chroma_cens, axis=1)
            std_val = np.std(chroma_cens, axis=1)
            skew_val = skew(chroma_cens, axis=1)
            kurtosis_val = kurtosis(chroma_cens, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "mfcc":
            min_val = np.min(mfcc, axis=1)
            mean_val = np.mean(mfcc, axis=1)
            median_val = np.median(mfcc, axis=1)
            max_val = np.max(mfcc, axis=1)
            std_val = np.std(mfcc, axis=1)
            skew_val = skew(mfcc, axis=1)
            kurtosis_val = kurtosis(mfcc, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "rms":
            min_val = np.min(rms, axis=1)
            mean_val = np.mean(rms, axis=1)
            median_val = np.median(rms, axis=1)
            max_val = np.max(rms, axis=1)
            std_val = np.std(rms, axis=1)
            skew_val = skew(rms, axis=1)
            kurtosis_val = kurtosis(rms, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "spectral_centroid":
            min_val = np.min(spectral_centroid, axis=1)
            mean_val = np.mean(spectral_centroid, axis=1)
            median_val = np.median(spectral_centroid, axis=1)
            max_val = np.max(spectral_centroid, axis=1)
            std_val = np.std(spectral_centroid, axis=1)
            skew_val = skew(spectral_centroid, axis=1)
            kurtosis_val = kurtosis(spectral_centroid, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "spectral_bandwidth":
            min_val = np.min(spectral_bandwidth, axis=1)
            mean_val = np.mean(spectral_bandwidth, axis=1)
            median_val = np.median(spectral_bandwidth, axis=1)
            max_val = np.max(spectral_bandwidth, axis=1)
            std_val = np.std(spectral_bandwidth, axis=1)
            skew_val = skew(spectral_bandwidth, axis=1)
            kurtosis_val = kurtosis(spectral_bandwidth, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "spectral_contrast":
            min_val = np.min(spectral_contrast, axis=1)
            mean_val = np.mean(spectral_contrast, axis=1)
            median_val = np.median(spectral_contrast, axis=1)
            max_val = np.max(spectral_contrast, axis=1)
            std_val = np.std(spectral_contrast, axis=1)
            skew_val = skew(spectral_contrast, axis=1)
            kurtosis_val = kurtosis(spectral_contrast, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "spectral_rolloff":
            min_val = np.min(spectral_rolloff, axis=1)
            mean_val = np.mean(spectral_rolloff, axis=1)
            median_val = np.median(spectral_rolloff, axis=1)
            max_val = np.max(spectral_rolloff, axis=1)
            std_val = np.std(spectral_rolloff, axis=1)
            skew_val = skew(spectral_rolloff, axis=1)
            kurtosis_val = kurtosis(spectral_rolloff, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])

        if feature == "tonnetz":
            fmin_val = np.min(tonnetz, axis=1)
            mean_val = np.mean(tonnetz, axis=1)
            median_val = np.median(tonnetz, axis=1)
            max_val = np.max(tonnetz, axis=1)
            std_val = np.std(tonnetz, axis=1)
            skew_val = skew(tonnetz, axis=1)
            kurtosis_val = kurtosis(tonnetz, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(37):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "zero_crossing_rate":
            min_val = np.min(zero_crossing_rate, axis=1)
            mean_val = np.mean(zero_crossing_rate, axis=1)
            median_val = np.median(zero_crossing_rate, axis=1)
            max_val = np.max(zero_crossing_rate, axis=1)
            std_val = np.std(zero_crossing_rate, axis=1)
            skew_val = skew(zero_crossing_rate, axis=1)
            kurtosis_val = kurtosis(zero_crossing_rate, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])

    
    
    df = pd.DataFrame(features)
    return df

def scale_reduction(df):
    scale = load("./scale.joblib")
    scaling = scale.transform(df)
    df = pd.DataFrame(scaling,columns=df.columns)
    pca = load("./PCA.joblib")
    data_reduced = pca.transform(df)
    df = pd.DataFrame(data_reduced,columns=[f"PCA{i}" for i in range(1,183)])
    return df
    
def prediction(df):
    model = load("./Naive_Bayes.joblib")
    pred = model.predict(df)
    if pred == 1:
        return "Popular"
    if pred == 0:
        return "NOT Popular"

def get_track_name(link):
    client_credentials_manager = SpotifyClientCredentials(client_id='7874bf62a03849a58ed846062fb22ea4',
                                                          client_secret='9b40f23f9ba3466f9a5a0bf3b2ad5a9a')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    track_info = sp.track(link)
    track_name = track_info['name']
    artists = ', '.join([artist['name'] for artist in track_info['artists']])
    return track_name, artists

def download_mp3(track_name, artists):
    ydl_opts = {
        'default_search': 'ytsearch',
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f"{input_file}"+f"{track_name}" + ".%(ext)s"
    }
    search_query = f"{track_name} {artists} official video"
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_query, download=True)

# Create your views here.
def home(request):
    return render(request, 'home.html')

@csrf_protect
def upload(request):    
    if request.method == 'POST' and request.FILES['audio_file']:
        song = request.FILES['audio_file']
        song_name = song.name
        extract_chorus(song_name)
        df = extract_features(song_name)
        scale_df = scale_reduction(df)
        pred = prediction(scale_df) 
        return render(request,'upload.html',{"pred": pred, "song_name":song_name})
    return render(request, 'upload.html')

@csrf_protect
def link(request):
    if request.method == 'POST':
        link = request.POST.get('link')
        track_name, artists = get_track_name(link)
        if f"{track_name}.mp3" not in os.listdir(input_file):
            download_mp3(track_name,artists)
        extract_chorus(f"{track_name}.mp3")
        df = extract_features(f"{track_name}.mp3")
        scale_df = scale_reduction(df)
        pred = prediction(scale_df)
        
        return render(request,'link.html',{"pred": pred, "track_name":f"{track_name}.mp3"})

    return render(request, 'link.html')
