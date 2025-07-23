from huggingface_hub import hf_hub_download
from json import load as json_load , dump as json_dump
from TTS.utils.synthesizer import Synthesizer

class Indic_TTS:
    def __init__(self,lang,device):
        model_id = "shethjenil/INDIC_TTS"
        config_path = hf_hub_download(model_id, lang+"_fastpitch_config.json")
        conf = json_load(open(config_path))
        conf['speakers_file'] = conf['model_args']['speakers_file'] = hf_hub_download(model_id, lang+"_fastpitch_speakers.pth")
        json_dump(conf, open(config_path, 'w'))
        self.synthesizer = Synthesizer(hf_hub_download(model_id, lang+"_fastpitch_best_model.pth"),config_path,vocoder_checkpoint=hf_hub_download(model_id, lang+"_hifigan_best_model.pth"),vocoder_config=hf_hub_download(model_id, lang+"_hifigan_config.json"),use_cuda= "cuda" in str(device))
        self.speakers = self.synthesizer.tts_model.speaker_manager.speaker_names
        self.full_name = {'as': 'Assamese', 'bn': 'Bangla', 'brx': 'Boro', 'en': 'Indian English', 'en+hi': 'Hinglish', 'gu': 'Gujarati', 'hi': 'Hindi', 'kn': 'Kannada', 'ml': 'Malayalam', 'mni': 'Manipuri', 'mr': 'Marathi', 'or': 'Oriya', 'pa': 'Panjabi', 'raj': 'Rajasthani', 'ta': 'Tamil', 'te': 'Telugu'}[lang]

    def predict(self,text:str,speaker:str,save_path = "output.wav"):
        self.synthesizer.save_wav(self.synthesizer.tts(text,speaker),save_path)
        return save_path
