import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pyximport
pyximport.install()

from indicai_projects.trans_processor import IndicProcessor

class IndicTrans:
    def __init__(self,en2indic_path,indic2en_path,indic2indic_path):
        self.all_lang = ['asm_Beng', 'awa_Deva', 'ben_Beng', 'bho_Deva', 'brx_Deva', 'doi_Deva', 'eng_Latn', 'gom_Deva', 'gon_Deva', 'guj_Gujr', 'hin_Deva', 'hne_Deva', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kha_Latn', 'lus_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'mni_Beng', 'mni_Mtei', 'npi_Deva', 'ory_Orya', 'pan_Guru', 'san_Deva', 'sat_Olck', 'snd_Arab', 'snd_Deva', 'tam_Taml', 'tel_Telu', 'urd_Arab', 'unr_Deva']
        self.ip = IndicProcessor(inference=True)
        self.indictrans_en2indic_tokenizer = AutoTokenizer.from_pretrained(en2indic_path, trust_remote_code=True)
        self.indictrans_en2indic_model = AutoModelForSeq2SeqLM.from_pretrained(en2indic_path, trust_remote_code=True)
        self.indictrans_indic2en_tokenizer = AutoTokenizer.from_pretrained(indic2en_path, trust_remote_code=True)
        self.indictrans_indic2en_model = AutoModelForSeq2SeqLM.from_pretrained(indic2en_path, trust_remote_code=True)
        self.indictrans_indic2indic_tokenizer = AutoTokenizer.from_pretrained(indic2indic_path, trust_remote_code=True)
        self.indictrans_indic2indic_model = AutoModelForSeq2SeqLM.from_pretrained(indic2indic_path, trust_remote_code=True)
    def _translate(self,model,tokenizer,input_list: list[str], source_lang: str, target_lang: str)->list[str]:
        with torch.inference_mode():
            outputs = model.generate(**tokenizer(self.ip.preprocess_batch(input_list, src_lang=source_lang, tgt_lang=target_lang, visualize=False),padding="longest",truncation=True,max_length=256,return_tensors="pt"), num_beams=5, num_return_sequences=1, max_length=256)
        with tokenizer.as_target_tokenizer():
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.ip.postprocess_batch(outputs, lang=target_lang)
    def predict(self,input: str, source_lang: str, target_lang: str):
        assert source_lang != target_lang and source_lang in self.all_lang and target_lang in self.all_lang
        if source_lang == "eng_Latn":
            return self._translate(self.indictrans_en2indic_model,self.indictrans_en2indic_tokenizer,[input],source_lang,target_lang)[0]
        elif target_lang == "eng_Latn":
            return self._translate(self.indictrans_indic2en_model,self.indictrans_indic2en_tokenizer,[input],source_lang,target_lang)[0]
        else:
            return self._translate(self.indictrans_indic2indic_model,self.indictrans_indic2indic_tokenizer,[input],source_lang,target_lang)[0]
