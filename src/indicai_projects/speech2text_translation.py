from transformers import SeamlessM4Tv2ForSpeechToText,SeamlessM4TTokenizer, SeamlessM4TFeatureExtractor
from numpy import array as np_array,float32 as np_float32
from pydub import AudioSegment

class INDIC_SEAMLESS:
    def __init__(self,device):
            self.seamless_model = SeamlessM4Tv2ForSpeechToText.from_pretrained("shethjenil/INDIC_SEAMLESS").to(device)
            self.seamless_processor = SeamlessM4TFeatureExtractor.from_pretrained("shethjenil/INDIC_SEAMLESS")
            self.seamless_tokenizer = SeamlessM4TTokenizer.from_pretrained("shethjenil/INDIC_SEAMLESS")
            self.lang_conf = {'Assamese': 'asm', 'Bengali': 'ben', 'Gujarati': 'guj', 'Hindi': 'hin', 'Kannada': 'kan', 'Malayalam': 'mal', 'Marathi': 'mar', 'Odia': 'ory', 'Punjabi': 'pan', 'Tamil': 'tam', 'Telugu': 'tel', 'Urdu': 'urd', 'English': 'eng'}
    def predict_batch(self,audio_paths, target_lang):
        return self.seamless_tokenizer.batch_decode(self.seamless_model.generate(**self.seamless_processor([np_array(AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000).get_array_of_samples(), dtype=np_float32) / 32768.0 for path in audio_paths], sampling_rate=16000, return_tensors="pt", padding=True).to("cpu"), tgt_lang=self.lang_conf[target_lang]), skip_special_tokens=True)
    def predict(self,audio_path, target_lang):
        return self.predict_batch([audio_path],target_lang)[0]
