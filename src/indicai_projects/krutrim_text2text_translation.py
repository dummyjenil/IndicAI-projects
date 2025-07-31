import os
from pathlib import Path
from typing import List, Tuple, Union, Dict
from huggingface_hub import snapshot_download
import regex as re
import sentencepiece as spm
from indicnlp.normalize import indic_normalize
from indicnlp.tokenize import indic_detokenize, indic_tokenize
from indicnlp.tokenize.sentence_tokenize import DELIM_PAT_NO_DANDA, sentence_split
from indicnlp.transliterate import unicode_transliterate
from mosestokenizer import MosesSentenceSplitter
from nltk.tokenize import sent_tokenize
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
import ctranslate2
"""
A dictionary mapping intended to normalize the numerals in Indic languages from 
native script to Roman script. This is done to ensure that the figures / numbers 
mentioned in native script are perfectly preserved during translation.
"""
INDIC_NUM_MAP = {
    "\u09e6": "0",
    "0": "0",
    "\u0ae6": "0",
    "\u0ce6": "0",
    "\u0966": "0",
    "\u0660": "0",
    "\uabf0": "0",
    "\u0b66": "0",
    "\u0a66": "0",
    "\u1c50": "0",
    "\u06f0": "0",
    "\u09e7": "1",
    "1": "1",
    "\u0ae7": "1",
    "\u0967": "1",
    "\u0ce7": "1",
    "\u06f1": "1",
    "\uabf1": "1",
    "\u0b67": "1",
    "\u0a67": "1",
    "\u1c51": "1",
    "\u0c67": "1",
    "\u09e8": "2",
    "2": "2",
    "\u0ae8": "2",
    "\u0968": "2",
    "\u0ce8": "2",
    "\u06f2": "2",
    "\uabf2": "2",
    "\u0b68": "2",
    "\u0a68": "2",
    "\u1c52": "2",
    "\u0c68": "2",
    "\u09e9": "3",
    "3": "3",
    "\u0ae9": "3",
    "\u0969": "3",
    "\u0ce9": "3",
    "\u06f3": "3",
    "\uabf3": "3",
    "\u0b69": "3",
    "\u0a69": "3",
    "\u1c53": "3",
    "\u0c69": "3",
    "\u09ea": "4",
    "4": "4",
    "\u0aea": "4",
    "\u096a": "4",
    "\u0cea": "4",
    "\u06f4": "4",
    "\uabf4": "4",
    "\u0b6a": "4",
    "\u0a6a": "4",
    "\u1c54": "4",
    "\u0c6a": "4",
    "\u09eb": "5",
    "5": "5",
    "\u0aeb": "5",
    "\u096b": "5",
    "\u0ceb": "5",
    "\u06f5": "5",
    "\uabf5": "5",
    "\u0b6b": "5",
    "\u0a6b": "5",
    "\u1c55": "5",
    "\u0c6b": "5",
    "\u09ec": "6",
    "6": "6",
    "\u0aec": "6",
    "\u096c": "6",
    "\u0cec": "6",
    "\u06f6": "6",
    "\uabf6": "6",
    "\u0b6c": "6",
    "\u0a6c": "6",
    "\u1c56": "6",
    "\u0c6c": "6",
    "\u09ed": "7",
    "7": "7",
    "\u0aed": "7",
    "\u096d": "7",
    "\u0ced": "7",
    "\u06f7": "7",
    "\uabf7": "7",
    "\u0b6d": "7",
    "\u0a6d": "7",
    "\u1c57": "7",
    "\u0c6d": "7",
    "\u09ee": "8",
    "8": "8",
    "\u0aee": "8",
    "\u096e": "8",
    "\u0cee": "8",
    "\u06f8": "8",
    "\uabf8": "8",
    "\u0b6e": "8",
    "\u0a6e": "8",
    "\u1c58": "8",
    "\u0c6e": "8",
    "\u09ef": "9",
    "9": "9",
    "\u0aef": "9",
    "\u096f": "9",
    "\u0cef": "9",
    "\u06f9": "9",
    "\uabf9": "9",
    "\u0b6f": "9",
    "\u0a6f": "9",
    "\u1c59": "9",
    "\u0c6f": "9",
}


URL_PATTERN = r'\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b'
EMAIL_PATTERN = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
# handles dates, time, percentages, proportion, ratio, etc
NUMERAL_PATTERN = r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)"
# handles upi, social media handles and hashtags
OTHER_PATTERN = r'[A-Za-z0-9]*[#|@]\w+'


def normalize_indic_numerals(line: str):
    """
    Normalize the numerals in Indic languages from native script to Roman script (if present).
    
    Args:
        line (str): an input string with Indic numerals to be normalized.
    
    Returns:
        str: an input string with the all Indic numerals normalized to Roman script.
    """
    return "".join([INDIC_NUM_MAP.get(c, c) for c in line])


def wrap_with_placeholders(text: str, patterns: list) -> Tuple[str, dict]:
    """
    Wraps substrings with matched patterns in the given text with placeholders and returns
    the modified text along with a mapping of the placeholders to their original value.
    
    Args:
        text (str): an input string which needs to be wrapped with the placeholders.
        pattern (list): list of patterns to search for in the input string.
    
    Returns:
        Tuple[str, dict]: a tuple containing the modified text and a dictionary mapping 
            placeholders to their original values.
    """
    serial_no = 1
    
    placeholder_entity_map = dict()
    
    for pattern in patterns:
        matches = set(re.findall(pattern, text))
        
        # wrap common match with placeholder tags
        for match in matches:
            if pattern==URL_PATTERN :
                #Avoids false positive URL matches for names with initials.
                temp = match.replace(".",'')
                if len(temp)<4:
                    continue
            if pattern==NUMERAL_PATTERN :
                #Short numeral patterns do not need placeholder based handling. e.g., masking list numberings might mess up the list structure
                temp = match.replace(" ",'').replace(".",'').replace(":",'')
                if len(temp)<2:
                    continue
            
            #Set of Translations of "ID" in all the suppported languages have been collated.            
            #This has been added to deal with edge cases where placeholders might get translated. 
            indic_failure_cases = ['آی ڈی ', 'ꯑꯥꯏꯗꯤ', 'आईडी', 'आई . डी . ', 'ऐटि', 'آئی ڈی ', 'ᱟᱭᱰᱤ ᱾', 'आयडी', 'ऐडि', 'आइडि']         
            placeholder = "<ID{}>".format(serial_no)
            alternate_placeholder = "< ID{} >".format(serial_no)                    
            placeholder_entity_map[placeholder] = match
            placeholder_entity_map[alternate_placeholder] = match
            
            for i in indic_failure_cases:
                placeholder_temp = "<{}{}>".format(i,serial_no)
                placeholder_entity_map[placeholder_temp] = match
                placeholder_temp = "< {}{} >".format(i, serial_no)
                placeholder_entity_map[placeholder_temp] = match
                placeholder_temp = "< {} {} >".format(i, serial_no)
                placeholder_entity_map[placeholder_temp] = match
            
            text = text.replace(match, placeholder)
            serial_no+=1
    
    text = re.sub("\s+", " ", text)
    
    #Regex has failure cases in trailing "/" in URLs, so this is a workaround. 
    text = text.replace(">/",">")
        
    return text, placeholder_entity_map


def normalize(text: str, patterns: list = [EMAIL_PATTERN, URL_PATTERN, NUMERAL_PATTERN, OTHER_PATTERN]) -> Tuple[str, dict]:
    """
    Normalizes and wraps the spans of input string with placeholder tags. It first normalizes
    the Indic numerals in the input string to Roman script. Later, it uses the input string with normalized
    Indic numerals to wrap the spans of text matching the pattern with placeholder tags.
    
    Args:
        text (str): input string.
        pattern (list): list of patterns to search for in the input string.
    
    Returns:
        Tuple[str, dict]: a tuple containing the modified text and a dictionary mapping 
            placeholders to their original values.
    """
    text = normalize_indic_numerals(text.strip("\n"))
    text, placeholder_entity_map  = wrap_with_placeholders(text, patterns)
    return text, placeholder_entity_map

"""
FLORES language code mapping to 2 letter ISO language code for compatibility 
with Indic NLP Library (https://github.com/anoopkunchukuttan/indic_nlp_library)
"""
flores_codes = {
    "asm_Beng": "as",
    "awa_Deva": "hi",
    "ben_Beng": "bn",
    "bho_Deva": "hi",
    "brx_Deva": "hi",
    "doi_Deva": "hi",
    "eng_Latn": "en",
    "gom_Deva": "kK",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "hne_Deva": "hi",
    "kan_Knda": "kn",
    "kas_Arab": "ur",
    "kas_Deva": "hi",
    "kha_Latn": "en",
    "lus_Latn": "en",
    "mag_Deva": "hi",
    "mai_Deva": "hi",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "bn",
    "mni_Mtei": "hi",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "hi",
    "sat_Olck": "or",
    "snd_Arab": "ur",
    "snd_Deva": "hi",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}


flores_to_iso = {
    "asm_Beng": "as",
    "awa_Deva": "awa",
    "ben_Beng": "bn",
    "bho_Deva": "bho",
    "brx_Deva": "brx",
    "doi_Deva": "doi",
    "eng_Latn": "en",
    "gom_Deva": "gom",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "hne_Deva": "hne",
    "kan_Knda": "kn",
    "kas_Arab": "ksa",
    "kas_Deva": "ksd",
    "kha_Latn": "kha",
    "lus_Latn": "lus",
    "mag_Deva": "mag",
    "mai_Deva": "mai",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "mnib",
    "mni_Mtei": "mnim",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "sa",
    "sat_Olck": "sat",
    "snd_Arab": "sda",
    "snd_Deva": "sdd",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}

iso_to_flores = {iso_code: flores_code for flores_code, iso_code in flores_to_iso.items()}
# Patch for digraphic langs.
iso_to_flores["ks"] = "kas_Arab"
iso_to_flores["ks_Deva"] = "kas_Deva"
iso_to_flores["mni"] = "mni_Mtei"
iso_to_flores["mni_Beng"] = "mni_Beng"
iso_to_flores["sd"] = "snd_Arab"
iso_to_flores["sd_Deva"] = "snd_Deva"

# IMPORTANT NOTE: DO NOT DIRECTLY EDIT THIS FILE
# This file was manually ported from `normalize-punctuation.perl`
# TODO: Only supports English, add others

multispace_regex = re.compile("[ ]{2,}")
multidots_regex = re.compile(r"\.{2,}")
end_bracket_space_punc_regex = re.compile(r"\) ([\.!:?;,])")
digit_space_percent = re.compile(r"(\d) %")
double_quot_punc = re.compile(r"\"([,\.]+)")
digit_nbsp_digit = re.compile(r"(\d) (\d)")

def punc_norm(text, lang="en"):
    text = text.replace('\r', '') \
                .replace('(', " (") \
                .replace(')', ") ") \
                \
                .replace("( ", "(") \
                .replace(" )", ")") \
                \
                .replace(" :", ':') \
                .replace(" ;", ';') \
                .replace('`', "'") \
                \
                .replace('„', '"') \
                .replace('“', '"') \
                .replace('”', '"') \
                .replace('–', '-') \
                .replace('—', " - ") \
                .replace('´', "'") \
                .replace('‘', "'") \
                .replace('‚', "'") \
                .replace('’', "'") \
                .replace("''", "\"") \
                .replace("´´", '"') \
                .replace('…', "...") \
                .replace(" « ", " \"") \
                .replace("« ", '"') \
                .replace('«', '"') \
                .replace(" » ", "\" ") \
                .replace(" »", '"') \
                .replace('»', '"') \
                .replace(" %", '%') \
                .replace("nº ", "nº ") \
                .replace(" :", ':') \
                .replace(" ºC", " ºC") \
                .replace(" cm", " cm") \
                .replace(" ?", '?') \
                .replace(" !", '!') \
                .replace(" ;", ';') \
                .replace(", ", ", ") \
                
    
    text = multispace_regex.sub(' ', text)
    text = multidots_regex.sub('.', text)
    text = end_bracket_space_punc_regex.sub(r")\1", text)
    text = digit_space_percent.sub(r"\1%", text)
    text = double_quot_punc.sub(r'\1"', text) # English "quotation," followed by comma, style
    text = digit_nbsp_digit.sub(r"\1.\2", text) # What does it mean?
    return text.strip(' ')



def split_sentences(paragraph: str, lang: str) -> List[str]:
    """
    Splits the input text paragraph into sentences. It uses `moses` for English and
    `indic-nlp` for Indic languages.

    Args:
        paragraph (str): input text paragraph.
        lang (str): flores language code.

    Returns:
        List[str] -> list of sentences.
    """
    if lang == "eng_Latn":
        with MosesSentenceSplitter(flores_codes[lang]) as splitter:
            sents_moses = splitter([paragraph])
        sents_nltk = sent_tokenize(paragraph)
        if len(sents_nltk) < len(sents_moses):
            sents = sents_nltk
        else:
            sents = sents_moses
        return [sent.replace("\xad", "") for sent in sents]
    else:
        return sentence_split(paragraph, lang=flores_codes[lang], delim_pat=DELIM_PAT_NO_DANDA)


def add_token(sent: str, src_lang: str, tgt_lang: str, delimiter: str = " ") -> str:
    """
    Add special tokens indicating source and target language to the start of the input sentence.
    The resulting string will have the format: "`{src_lang} {tgt_lang} {input_sentence}`".

    Args:
        sent (str): input sentence to be translated.
        src_lang (str): flores lang code of the input sentence.
        tgt_lang (str): flores lang code in which the input sentence will be translated.
        delimiter (str): separator to add between language tags and input sentence (default: " ").

    Returns:
        str: input sentence with the special tokens added to the start.
    """
    return src_lang + delimiter + tgt_lang + delimiter + sent


def apply_lang_tags(sents: List[str], src_lang: str, tgt_lang: str) -> List[str]:
    """
    Add special tokens indicating source and target language to the start of the each input sentence.
    Each resulting input sentence will have the format: "`{src_lang} {tgt_lang} {input_sentence}`".

    Args:
        sent (str): input sentence to be translated.
        src_lang (str): flores lang code of the input sentence.
        tgt_lang (str): flores lang code in which the input sentence will be translated.

    Returns:
        List[str]: list of input sentences with the special tokens added to the start.
    """
    tagged_sents = []
    for sent in sents:
        tagged_sent = add_token(sent.strip(), src_lang, tgt_lang)
        tagged_sents.append(tagged_sent)
    return tagged_sents


def truncate_long_sentences(
    sents: List[str], placeholder_entity_map_sents: List[Dict]
) -> Tuple[List[str], List[Dict]]:
    """
    Truncates the sentences that exceed the maximum sequence length.
    The maximum sequence for the IndicTrans2 model is limited to 256 tokens.

    Args:
        sents (List[str]): list of input sentences to truncate.

    Returns:
        Tuple[List[str], List[Dict]]: tuple containing the list of sentences with truncation applied and the updated placeholder entity maps.
    """
    MAX_SEQ_LEN = 1024
    new_sents = []
    placeholders = []

    for j, sent in enumerate(sents):
        words = sent.split()
        num_words = len(words)
        if num_words > MAX_SEQ_LEN:
            sents = []
            i = 0
            while i <= len(words):
                sents.append(" ".join(words[i : i + MAX_SEQ_LEN]))
                i += MAX_SEQ_LEN
            placeholders.extend([placeholder_entity_map_sents[j]] * (len(sents)))
            new_sents.extend(sents)
        else:
            placeholders.append(placeholder_entity_map_sents[j])
            new_sents.append(sent)
    return new_sents, placeholders


class KrutrimTrans:

    def __init__(
        self,
        ckpt_dir: str,
        device: str = "cpu",
        input_lang_code_format: str = "flores",
    ):
        """
        Initialize the model class.

        Args:
            ckpt_dir (str): path of the model checkpoint directory.
            device (str, optional): where to load the model (defaults: cuda).
        """
        self.en_tok = MosesTokenizer(lang="en")
        self.en_normalizer = MosesPunctNormalizer()
        self.en_detok = MosesDetokenizer(lang="en")
        self.xliterator = unicode_transliterate.UnicodeIndicTransliterator()
        self.sp_src = spm.SentencePieceProcessor(model_file=str(ckpt_dir/"model.SRC"))
        self.sp_tgt = spm.SentencePieceProcessor(model_file=str(ckpt_dir/"model.TGT"))

        self.input_lang_code_format = input_lang_code_format

        self.translator = ctranslate2.Translator(str(ckpt_dir), device=device, compute_type="auto")

    def translate_lines(self, lines: List[str], beam_len:int) -> List[str]:
        tokenized_sents = [x.strip().split(" ") for x in lines]
        translations = self.translator.translate_batch(
            tokenized_sents,
            max_batch_size=9216,
            batch_type="tokens",
            max_input_length=4096,
            max_decoding_length=4096,
            beam_size=beam_len,
        )
        translations = [" ".join(x.hypotheses[0]) for x in translations]
        return translations

    def paragraphs_batch_translate__multilingual(self, batch_payloads: List[tuple]) -> List[str]:
        """
        Translates a batch of input paragraphs (including pre/post processing)
        from any language to any language.

        Args:
            batch_payloads (List[tuple]): batch of long input-texts to be translated, each in format: (paragraph, src_lang, tgt_lang)

        Returns:
            List[str]: batch of paragraph-translations in the respective languages.
        """
        paragraph_id_to_sentence_range = []
        global__sents = []
        global__preprocessed_sents = []
        global__preprocessed_sents_placeholder_entity_map = []

        for i in range(len(batch_payloads)):
            paragraph, src_lang, tgt_lang = batch_payloads[i]
            if self.input_lang_code_format == "iso":
                src_lang, tgt_lang = iso_to_flores[src_lang], iso_to_flores[tgt_lang]

            batch = split_sentences(paragraph, src_lang)
            global__sents.extend(batch)

            preprocessed_sents, placeholder_entity_map_sents = self.preprocess_batch(
                batch, src_lang, tgt_lang
            )

            global_sentence_start_index = len(global__preprocessed_sents)
            global__preprocessed_sents.extend(preprocessed_sents)
            global__preprocessed_sents_placeholder_entity_map.extend(placeholder_entity_map_sents)
            paragraph_id_to_sentence_range.append(
                (global_sentence_start_index, len(global__preprocessed_sents))
            )

        translations = self.translate_lines(global__preprocessed_sents)

        translated_paragraphs = []
        for paragraph_id, sentence_range in enumerate(paragraph_id_to_sentence_range):
            tgt_lang = batch_payloads[paragraph_id][2]
            if self.input_lang_code_format == "iso":
                tgt_lang = iso_to_flores[tgt_lang]

            postprocessed_sents = self.postprocess(
                translations[sentence_range[0] : sentence_range[1]],
                global__preprocessed_sents_placeholder_entity_map[
                    sentence_range[0] : sentence_range[1]
                ],
                tgt_lang,
            )
            translated_paragraph = " ".join(postprocessed_sents)
            translated_paragraphs.append(translated_paragraph)

        return translated_paragraphs

    # translate a batch of sentences from src_lang to tgt_lang
    def batch_translate(self, batch: List[str], src_lang: str, tgt_lang: str, beam_len:int=3) -> List[str]:
        """
        Translates a batch of input sentences (including pre/post processing)
        from source language to target language.

        Args:
            batch (List[str]): batch of input sentences to be translated.
            src_lang (str): flores source language code.
            tgt_lang (str): flores target language code.

        Returns:
            List[str]: batch of translated-sentences generated by the model.
        """

        assert isinstance(batch, list)

        if self.input_lang_code_format == "iso":
            src_lang, tgt_lang = iso_to_flores[src_lang], iso_to_flores[tgt_lang]

        preprocessed_sents, placeholder_entity_map_sents = self.preprocess_batch(
            batch, src_lang, tgt_lang
        )

        translations = self.translate_lines(preprocessed_sents, beam_len)
        return self.postprocess(translations, placeholder_entity_map_sents, tgt_lang)

    # translate a paragraph from src_lang to tgt_lang
    def translate_paragraph(self, paragraph: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translates an input text paragraph (including pre/post processing)
        from source language to target language.

        Args:
            paragraph (str): input text paragraph to be translated.
            src_lang (str): flores source language code.
            tgt_lang (str): flores target language code.

        Returns:
            str: paragraph translation generated by the model.
        """

        assert isinstance(paragraph, str)

        if self.input_lang_code_format == "iso":
            flores_src_lang = iso_to_flores[src_lang]
        else:
            flores_src_lang = src_lang

        sents = split_sentences(paragraph, flores_src_lang)
        postprocessed_sents = self.batch_translate(sents, src_lang, tgt_lang)
        translated_paragraph = " ".join(postprocessed_sents)

        return translated_paragraph

    def preprocess_batch(self, batch: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        Preprocess an array of sentences by normalizing, tokenization, and possibly transliterating it. It also tokenizes the
        normalized text sequences using sentence piece tokenizer and also adds language tags.

        Args:
            batch (List[str]): input list of sentences to preprocess.
            src_lang (str): flores language code of the input text sentences.
            tgt_lang (str): flores language code of the output text sentences.

        Returns:
            Tuple[List[str], List[Dict]]: a tuple of list of preprocessed input text sentences and also a corresponding list of dictionary
                mapping placeholders to their original values.
        """
        preprocessed_sents, placeholder_entity_map_sents = self.preprocess(batch, lang=src_lang)
        tokenized_sents = self.apply_spm(preprocessed_sents)
        tokenized_sents, placeholder_entity_map_sents = truncate_long_sentences(
            tokenized_sents, placeholder_entity_map_sents
        )
        tagged_sents = apply_lang_tags(tokenized_sents, src_lang, tgt_lang)
        return tagged_sents, placeholder_entity_map_sents

    def apply_spm(self, sents: List[str]) -> List[str]:
        """
        Applies sentence piece encoding to the batch of input sentences.

        Args:
            sents (List[str]): batch of the input sentences.

        Returns:
            List[str]: batch of encoded sentences with sentence piece model
        """
        return [" ".join(self.sp_src.encode(sent, out_type=str)) for sent in sents]

    def preprocess_sent(
        self,
        sent: str,
        normalizer: Union[MosesPunctNormalizer, indic_normalize.IndicNormalizerFactory],
        lang: str,
    ) -> Tuple[str, Dict]:
        """
        Preprocess an input text sentence by normalizing, tokenization, and possibly transliterating it.

        Args:
            sent (str): input text sentence to preprocess.
            normalizer (Union[MosesPunctNormalizer, indic_normalize.IndicNormalizerFactory]): an object that performs normalization on the text.
            lang (str): flores language code of the input text sentence.

        Returns:
            Tuple[str, Dict]: A tuple containing the preprocessed input text sentence and a corresponding dictionary
            mapping placeholders to their original values.
        """
        iso_lang = flores_codes[lang]
        sent = punc_norm(sent, iso_lang)
        sent, placeholder_entity_map = normalize(sent)

        transliterate = True
        if lang.split("_")[1] in ["Arab", "Aran", "Olck", "Mtei", "Latn"]:
            transliterate = False

        if iso_lang == "en":
            processed_sent = " ".join(
                self.en_tok.tokenize(self.en_normalizer.normalize(sent.strip()), escape=False)
            )
        elif transliterate:
            # transliterates from the any specific language to devanagari
            # which is why we specify lang2_code as "hi".
            processed_sent = self.xliterator.transliterate(
                " ".join(
                    indic_tokenize.trivial_tokenize(normalizer.normalize(sent.strip()), iso_lang)
                ),
                iso_lang,
                "hi",
            ).replace(" ् ", "्")
        else:
            # we only need to transliterate for joint training
            processed_sent = " ".join(
                indic_tokenize.trivial_tokenize(normalizer.normalize(sent.strip()), iso_lang)
            )

        return processed_sent, placeholder_entity_map

    def preprocess(self, sents: List[str], lang: str):
        """
        Preprocess an array of sentences by normalizing, tokenization, and possibly transliterating it.

        Args:
            batch (List[str]): input list of sentences to preprocess.
            lang (str): flores language code of the input text sentences.

        Returns:
            Tuple[List[str], List[Dict]]: a tuple of list of preprocessed input text sentences and also a corresponding list of dictionary
                mapping placeholders to their original values.
        """
        processed_sents, placeholder_entity_map_sents = [], []

        if lang == "eng_Latn":
            normalizer = None
        else:
            normfactory = indic_normalize.IndicNormalizerFactory()
            normalizer = normfactory.get_normalizer(flores_codes[lang])

        for sent in sents:
            sent, placeholder_entity_map = self.preprocess_sent(sent, normalizer, lang)
            processed_sents.append(sent)
            placeholder_entity_map_sents.append(placeholder_entity_map)

        return processed_sents, placeholder_entity_map_sents

    def postprocess(
        self,
        sents: List[str],
        placeholder_entity_map: List[Dict],
        lang: str,
        common_lang: str = "hin_Deva",
    ) -> List[str]:
        """
        Postprocesses a batch of input sentences after the translation generations.

        Args:
            sents (List[str]): batch of translated sentences to postprocess.
            placeholder_entity_map (List[Dict]): dictionary mapping placeholders to the original entity values.
            lang (str): flores language code of the input sentences.
            common_lang (str, optional): flores language code of the transliterated language (defaults: hin_Deva).

        Returns:
            List[str]: postprocessed batch of input sentences.
        """

        lang_code, script_code = lang.split("_")
        # SPM decode
        for i in range(len(sents)):
            # sent_tokens = sents[i].split(" ")
            # sents[i] = self.sp_tgt.decode(sent_tokens)

            sents[i] = sents[i].replace(" ", "").replace("▁", " ").strip()

            # Fixes for Perso-Arabic scripts
            # TODO: Move these normalizations inside indic-nlp-library
            if script_code in {"Arab", "Aran"}:
                # UrduHack adds space before punctuations. Since the model was trained without fixing this issue, let's fix it now
                sents[i] = sents[i].replace(" ؟", "؟").replace(" ۔", "۔").replace(" ،", "،")
                # Kashmiri bugfix for palatalization: https://github.com/AI4Bharat/IndicTrans2/issues/11
                sents[i] = sents[i].replace("ٮ۪", "ؠ")

        assert len(sents) == len(placeholder_entity_map)

        for i in range(0, len(sents)):
            for key in placeholder_entity_map[i].keys():
                sents[i] = sents[i].replace(key, placeholder_entity_map[i][key])

        # Detokenize and transliterate to native scripts if applicable
        postprocessed_sents = []

        if lang == "eng_Latn":
            for sent in sents:
                postprocessed_sents.append(self.en_detok.detokenize(sent.split(" ")))
        else:
            for sent in sents:
                outstr = indic_detokenize.trivial_detokenize(
                    self.xliterator.transliterate(
                        sent, flores_codes[common_lang], flores_codes[lang]
                    ),
                    flores_codes[lang],
                )
                
                # Oriya bug: indic-nlp-library produces ଯ଼ instead of ୟ when converting from Devanagari to Odia
                # TODO: Find out what's the issue with unicode transliterator for Oriya and fix it
                if lang_code == "ory":
                    outstr = outstr.replace("ଯ଼", 'ୟ')

                postprocessed_sents.append(outstr)

        return postprocessed_sents



class IndicTrans:
    def __init__(self,device = "cpu",model_path=None):
        if not model_path:
            model_path = snapshot_download("shethjenil/KrutrimTrans")
        model_path = Path(model_path)
        self.en2indic_model = KrutrimTrans(model_path/"en2indic",device)
        self.indic2en_model = KrutrimTrans(model_path/"indic2en",device)
        self.lang_conf = {'english':'eng_Latn', 'bengali': 'ben_Beng', 'hindi': 'hin_Deva', 'kannada': 'kan_Knda', 'marathi': 'mar_Deva', 'malayalam': 'mal_Mlym', 'gujarati': 'guj_Gujr', 'punjabi': 'pan_Guru', 'telugu': 'tel_Telu', 'tamil': 'tam_Taml'}
        self.all_lang = list(self.lang_conf.keys())

    def predict(self,input: str, source_lang: str, target_lang: str):
        assert source_lang != target_lang and source_lang in self.all_lang and target_lang in self.all_lang
        if source_lang == "english":
            return self.en2indic_model.batch_translate([input],self.lang_conf[source_lang],self.lang_conf[target_lang])[0]
        elif target_lang == "english":
            return self.indic2en_model.batch_translate([input],self.lang_conf[source_lang],self.lang_conf[target_lang])[0]
        else:
            return self.en2indic_model.batch_translate([self.indic2en_model.batch_translate([input],self.lang_conf[source_lang],self.lang_conf["english"])[0]],self.lang_conf["english"],self.lang_conf[target_lang])[0]

    def batch_predict(self,inputs: List[str], source_lang: str, target_lang: str):
        assert source_lang != target_lang and source_lang in self.all_lang and target_lang in self.all_lang
        if source_lang == "english":
            return self.en2indic_model.batch_translate(inputs,self.lang_conf[source_lang],self.lang_conf[target_lang])
        elif target_lang == "english":
            return self.indic2en_model.batch_translate(inputs,self.lang_conf[source_lang],self.lang_conf[target_lang])
        else:
            return self.en2indic_model.batch_translate(self.indic2en_model.batch_translate(inputs,self.lang_conf[source_lang],self.lang_conf["english"]),self.lang_conf["english"],self.lang_conf[target_lang])
