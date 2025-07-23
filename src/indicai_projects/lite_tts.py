import soundfile as sf
from transformers import AutoModel, AutoTokenizer
import torch

class Lite_TTS:
    def __init__(self,device):
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/vits_rasa_13", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("ai4bharat/vits_rasa_13", trust_remote_code=True)
        self.model.to(device)
        self.speakers = ['ASM_F', 'ASM_M', 'BEN_F', 'BEN_M', 'BRX_F', 'BRX_M', 'DOI_F', 'DOI_M', 'KAN_F', 'KAN_M', 'MAI_M', 'MAL_F', 'MAR_F', 'MAR_M', 'NEP_F', 'PAN_F', 'PAN_M', 'SAN_M', 'TAM_F', 'TEL_F']
        self.styles = ['ALEXA', 'ANGER', 'BB', 'BOOK', 'CONV', 'DIGI', 'DISGUST', 'FEAR', 'HAPPY', 'NEWS', 'SAD', 'SURPRISE', 'UMANG', 'WIKI']
    def predict(self,text,speaker,style,output_file="audio.wav"):
        with torch.no_grad():
            sf.write(output_file, self.model(self.tokenizer(text,return_tensors="pt")['input_ids'], speaker_id=self.speakers.index(speaker), emotion_id=self.styles.index(style)).waveform.squeeze(), self.model.config.sampling_rate)
        return output_file