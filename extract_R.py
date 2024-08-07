import os
import ssl

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from playsound import playsound
import soundfile
import torch
from torch import nn
import torch.utils
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, logging
import tqdm


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
    

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, audios, labels):
        self.audios = audios
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        input_ids = torch.tensor(self.audios[index]).squeeze()
        target_ids = torch.tensor(self.labels[index]).squeeze()
        return {"input_ids": input_ids, "labels": target_ids}

def extract_R():
    logging.set_verbosity_error()
    ssl._create_default_https_context = ssl._create_unverified_context
    PATH = "train_audio"
    LETTER_LENGTH = 0.2
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    PRETRAINED_MODEL_SAMPLE_RATE = 16000
    LETTER_R_ORD = 23
    MODEL_WINDOW_SIZE = 320

    device = torch.device("mps")
    processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model: Wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)

    labeled_dataset = pd.read_csv("train_gt (1).csv")
    for file in tqdm.tqdm(os.listdir(PATH)):
        audio_label = bool(labeled_dataset[labeled_dataset["Filename"] == file]["Label"].values[0])
        if os.path.exists(f"{'burr_audio' if audio_label else 'normal_audio'}/{file}_R0.wav"):
            continue
        audio, sr = librosa.load(os.path.join(PATH, file))
        audio = torchaudio.functional.resample(torch.tensor(audio), sr, PRETRAINED_MODEL_SAMPLE_RATE)
        #print("burr" if audio_label else "normal")
        letter_size = round(LETTER_LENGTH * sr)
        if audio.size()[0] % letter_size != 0:
            audio = audio[: -int(audio.size()[0] % letter_size)]# for equal split, otherwise np raises error
        data = SpeechDataset(audio, torch.tensor([audio_label] * audio.size()[0]))
        inputs = processor(data.audios, 
                           sampling_rate=PRETRAINED_MODEL_SAMPLE_RATE, 
                           return_tensors="pt", 
                           padding=True
                           ).to(device)
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        # predicted_ids = torch.argmax(logits, dim=-1)
        predicted_ids_along_time = torch.argmax(logits.squeeze(0), dim=1)
        R_pos = torch.nonzero(predicted_ids_along_time == LETTER_R_ORD).squeeze(1)
        for i, r_pos in enumerate(R_pos):
            r_pos = r_pos.item() * MODEL_WINDOW_SIZE
            R_data = audio[r_pos - letter_size // 2 : r_pos + letter_size // 2]
            soundfile.write(
                f"{'burr_audio' if audio_label else 'normal_audio'}/{file}_R{i}.wav", 
                R_data, 
                samplerate=PRETRAINED_MODEL_SAMPLE_RATE
            )
        # predicted_sentences = processor.batch_decode(predicted_ids) 
        # print("Predicted:", predicted_sentences)
        # # plt.imshow(logits[0].cpu().T, interpolation="nearest")
        # # plt.show()
        # repeat = True
        # while repeat:
        #     playsound(os.path.join(PATH, file), block=True)
        #     repeat = input("Repeat? (y/n)")
        #     while repeat not in ("y", "n"):
        #         repeat = input("Invalid input. Valid inputs are: 'y', 'n'")
        #     repeat = repeat == "y"
        

if __name__ == "__main__":
    extract_R()