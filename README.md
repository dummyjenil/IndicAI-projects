```bash
IndicAI-projects[lang_detection,text2text_translation,transliteration_rnn,transliteration_transformer]
```

```python
import gradio as gr
from os import getenv
from huggingface_hub import hf_hub_download
from torch import device as Device
from torch.cuda import is_available as cuda_is_available
from transformers import AutoTokenizer
from indicai_projects.lang_detection import IndicLangDet
from indicai_projects.text2text_translation import IndicTrans
from indicai_projects.transliteration_rnn import Transliteration_RNN , rnn_conf
from indicai_projects.transliteration_transformer import Transliteration_Transformer

en2indic_rnn_lang = getenv("en2indic_rnn_lang","hi")
en2indic_lang = getenv("en2indic_lang","hi")
device = Device("cuda" if cuda_is_available() else "cpu")
LID_model = IndicLangDet(hf_hub_download("ai4bharat/IndicLID-BERT","basline_nn_simple.pt"),hf_hub_download("ai4bharat/IndicLID-FTR","model_baseline_roman.bin"),hf_hub_download("ai4bharat/IndicLID-FTN","model_baseline_roman.bin"),AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only"),device)
en2indic_RNN_model = Transliteration_RNN(hf_hub_download("shethjenil/Indic-Transliteration-RNN", rnn_conf[en2indic_rnn_lang]["weight"]) ,hf_hub_download("shethjenil/Indic-Transliteration-RNN", rnn_conf[en2indic_rnn_lang]["script"]),hf_hub_download("shethjenil/Indic-Transliteration-RNN", rnn_conf[en2indic_rnn_lang]["vocab"]),device)
en2indic_model = Transliteration_Transformer({en2indic_lang:hf_hub_download("shethjenil/Indic-Transliteration-Word-Prob-Dicts",f"{en2indic_lang}_word_prob_dict.json")},hf_hub_download("ai4bharat/IndicXlit","indicxlit-en-indic-v1.0/transformer/indicxlit.pt"),hf_hub_download("shethjenil/Indic-Transliteration-Word-Prob-Dicts", "corpus.zip"),{en2indic_lang},device)
indic2en_model = Transliteration_Transformer({"en":hf_hub_download("shethjenil/Indic-Transliteration-Word-Prob-Dicts","en_word_prob_dict.json")},hf_hub_download("ai4bharat/IndicXlit","indicxlit-indic-en-v1.0/transformer/indicxlit.pt"),hf_hub_download("shethjenil/Indic-Transliteration-Word-Prob-Dicts", "corpus.zip"),{"en"},device)
indic_trans_model = IndicTrans("prajdabre/rotary-indictrans2-en-indic-dist-200M","prajdabre/rotary-indictrans2-indic-en-dist-200M","ai4bharat/indictrans2-indic-indic-dist-320M")

gr.TabbedInterface(
    [
        gr.Interface(LID_model.predict,gr.Textbox(label="Enter text"),[gr.Textbox(label="Language"), gr.Number(label="Accuracy in %")],title="Language Detection",),
        gr.Interface(en2indic_RNN_model.predict,[gr.Textbox(label="Enter Word"),gr.Number(label="Enter Variation Number", value=1),],gr.List(label="Transliteration Result"),title="RNN Transliteration",),
        gr.Interface(lambda word, topk: en2indic_model._transliterate_word(word, "en", en2indic_lang, topk, nativize_numerals=True),[gr.Textbox(label="Enter Word"),gr.Number(label="Enter Variation Number", value=1),],gr.List(label="Transliteration Result"),title=f"En2Indic Transliteration",),
        gr.Interface(lambda word, topk, indic2en_lang: indic2en_model._transliterate_word(word, indic2en_lang, "en", topk, nativize_numerals=True),[gr.Textbox(label="Enter Word"),gr.Number(label="Enter Variation Number", value=1),gr.Dropdown(indic2en_model.all_supported_langs,label="input lang")],gr.List(label="Transliteration Result"),title="Indic2En Transliteration",),
        gr.Interface(indic_trans_model.predict,[gr.Textbox(label="Input Text"),gr.Dropdown(indic_trans_model.all_lang, label="Source Language"),gr.Dropdown(indic_trans_model.all_lang, label="Target Language")],gr.Textbox(label="Result")),
    ],
    [
        "Language Detection",
        f"RNN en2{en2indic_rnn_lang} Transliteration",
        f"TRANSFORMER en2{en2indic_lang} Transliteration",
        "Indic2en Transliteration",
        "Text Translatation"
    ],
).launch()

```

```bash
IndicAI-projects[indic_tts,sanskrit_tts,lite_tts,speech2text_translation,speech2text_all]
```

```python

import gradio as gr
from os import getenv
from huggingface_hub import hf_hub_download
from torch import device as Device
from torch.cuda import is_available as cuda_is_available
from indicai_projects.indic_tts import Indic_TTS
from indicai_projects.sanskrit_tts import SansTTS
from indicai_projects.lite_tts import Lite_TTS
from indicai_projects.speech2text_translation import INDIC_SEAMLESS
from indicai_projects.speech2text_all import Indic_STT_ALL
from zipfile import ZipFile
device = Device("cuda" if cuda_is_available() else "cpu")

indic_tts_lang = getenv("indic_tts_lang","hi")
ZipFile(hf_hub_download("shethjenil/CONFORMER_INDIC_STT","conformer_onnx.zip"), 'r').extractall("conformer_onnx")
indic_stt_all_model = Indic_STT_ALL("conformer_onnx",device)
indic_tts = Indic_TTS(indic_tts_lang,device)
sans_tts_model = SansTTS(hf_hub_download("shethjenil/INDIC_TTS","sanskrit_tts_model.pth"),device)
vits_tts = Lite_TTS(device)
indic_seamless_model = INDIC_SEAMLESS(device)

gr.TabbedInterface(
    [
        gr.Interface(indic_tts.predict,[gr.Textbox(label="Enter Text"),gr.Dropdown(indic_tts.speakers, label="speaker"),],gr.Audio(type="filepath", label="Speech")),
        gr.Interface(sans_tts_model.predict,[gr.Textbox(value="उद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः"),gr.Dropdown(sans_tts_model.speakers,label='Speaker',type='index'),gr.Slider(0.5,2,1,step=0.1,label='Speaking Speed')],gr.Audio(label="Speech")),
        gr.Interface(vits_tts.predict,[gr.Textbox(),gr.Dropdown(vits_tts.speakers,label='Speaker'),gr.Dropdown(vits_tts.styles,label='Style')],gr.Audio(label="Speech")),
        gr.Interface(indic_seamless_model.predict,[gr.Audio(type="filepath"),gr.Dropdown(list(indic_seamless_model.lang_conf.keys()), label="Target Language"),],gr.Text(label="Translations"),title="Audio Translation",),
        gr.Interface(indic_stt_all_model.predict,[gr.Audio(type="filepath"),gr.Dropdown(indic_stt_all_model.supported_langs,label='Language')],[gr.Text(label="CTC"),gr.Text(label="RNNT")]),
    ],
    [
        f"{indic_tts.full_name} TTS",
        "Sanskrit TTS",
        "Lite TTS With ",
        "Audio Translation",
        "All Indic Speech To Text",
    ],
).launch()

```