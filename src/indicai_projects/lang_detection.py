from re import compile as re_compile,findall as re_findall
from pandas import DataFrame
from torch import device as Device , load as torch_load,no_grad as torch_no_grad,max as torch_max
from torch.utils.data import Dataset,DataLoader
from fasttext import load_model as fasttext_load_model

class IndicBERT_Data(Dataset):
    def __init__(self, indices, X):
        self.size = len(X)
        self.x = X
        self.i = indices
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return (self.i[idx], self.x[idx])

class IndicLangDet:
    def __init__(self,bert_path:str,ftr_path:str,ftn_path:str,tokenizer,device:Device,input_threshold = 0.5, roman_lid_threshold = 0.6):
        self.device = device
        langs = ['asm_Latn', 'ben_Latn', 'brx_Latn', 'guj_Latn', 'hin_Latn', 'kan_Latn', 'kas_Latn', 'kok_Latn', 'mai_Latn', 'mal_Latn', 'mni_Latn', 'mar_Latn', 'nep_Latn', 'ori_Latn', 'pan_Latn', 'san_Latn', 'snd_Latn', 'tam_Latn', 'tel_Latn', 'urd_Latn', 'eng_Latn', 'other', 'asm_Beng', 'ben_Beng', 'brx_Deva', 'doi_Deva', 'guj_Gujr', 'hin_Deva', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kok_Deva', 'mai_Deva', 'mal_Mlym', 'mni_Beng', 'mni_Meti', 'mar_Deva', 'nep_Deva', 'ori_Orya', 'pan_Guru', 'san_Deva', 'sat_Olch', 'snd_Arab', 'tam_Tamil', 'tel_Telu', 'urd_Arab']
        self.IndicLID_FTN = fasttext_load_model(ftn_path)
        self.IndicLID_FTR = fasttext_load_model(ftr_path)
        self.IndicLID_BERT = torch_load(bert_path, map_location = self.device,weights_only=False)
        self.IndicLID_BERT.eval()
        self.IndicLID_BERT_tokenizer = tokenizer
        self.input_threshold = input_threshold
        self.model_threshold = roman_lid_threshold
        self.classes = 47
        self.IndicLID_lang_code_dict = {cont:ind for ind,cont in enumerate(langs)}
        self.IndicLID_lang_code_dict_reverse = {ind:cont for ind,cont in enumerate(langs)}

    def char_percent_check(self, input):
        total_chars = len(list(input)) - (len(re_compile('[@_!#$%^&*()<>?/\|}{~:]').findall(input)) + len(re_findall('\s', input)) + len(re_findall('\n', input)))
        if total_chars == 0:
            return 0
        return len(re_compile('[a-zA-Z0-9]').findall(input))/total_chars

    def native_inference(self, input_list, output_dict):
        if not input_list:
            return output_dict
        input_texts = [line[1] for line in input_list]
        IndicLID_FTN_predictions = self.IndicLID_FTN.predict(input_texts)
        for input, pred_label, pred_score in zip(input_list, IndicLID_FTN_predictions[0], IndicLID_FTN_predictions[1]):
            output_dict[input[0]] = (input[1], pred_label[0][9:], pred_score[0], 'IndicLID-FTN')
        return output_dict

    def roman_inference(self, input_list, output_dict, batch_size):
        if not input_list:
            return output_dict
        input_texts = [line[1] for line in input_list]
        IndicLID_FTR_predictions = self.IndicLID_FTR.predict(input_texts)
        IndicLID_BERT_inputs = []
        for input, pred_label, pred_score in zip(input_list, IndicLID_FTR_predictions[0], IndicLID_FTR_predictions[1]):
            if pred_score[0] > self.model_threshold:
                output_dict[input[0]] = (input[1], pred_label[0][9:], pred_score[0], 'IndicLID-FTR')
            else:
                IndicLID_BERT_inputs.append(input)
        return self.IndicBERT_roman_inference(IndicLID_BERT_inputs, output_dict, batch_size)

    def IndicBERT_roman_inference(self, IndicLID_BERT_inputs, output_dict, batch_size):
        if not IndicLID_BERT_inputs:
            return output_dict
        df = DataFrame(IndicLID_BERT_inputs)
        dataloader = self.get_dataloaders(df.iloc[:,0], df.iloc[:,1], batch_size)
        with torch_no_grad():
            for data in dataloader:
                batch_indices = data[0]
                batch_inputs = data[1]
                word_embeddings = self.IndicLID_BERT_tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
                word_embeddings = word_embeddings.to(self.device)
                batch_outputs = self.IndicLID_BERT(word_embeddings['input_ids'], token_type_ids=word_embeddings['token_type_ids'], attention_mask=word_embeddings['attention_mask'])
                _, batch_predicted = torch_max(batch_outputs.logits, 1)
                for index, input, pred_label, logit in zip(batch_indices, batch_inputs, batch_predicted, batch_outputs.logits):
                    output_dict[index] = (input,self.IndicLID_lang_code_dict_reverse[pred_label.item()],logit[pred_label.item()].item(), 'IndicLID-BERT')
        return output_dict
    
    def post_process(self, output_dict:dict):
        return [output_dict[index] for index in sorted(list(output_dict.keys()))]
    def get_dataloaders(self, indices, input_texts, batch_size):
        return DataLoader(IndicBERT_Data(indices, input_texts),batch_size=batch_size,shuffle=False)
    def lang_detection(self, input):
        return self.batch_predict([input], 1)[0]
    def batch_predict(self, input_list, batch_size):
        output_dict = {}
        roman_inputs = []
        native_inputs = []
        for index, input in enumerate(input_list):
            if self.char_percent_check(input) > self.input_threshold:
                roman_inputs.append((index, input))
            else:
                native_inputs.append((index, input))
        return self.post_process(self.roman_inference(roman_inputs, self.native_inference(native_inputs, output_dict), batch_size))
    def predict(self,input_text):
        output = self.lang_detection(input_text)
        return output[1],round(float(output[2]),2)*100