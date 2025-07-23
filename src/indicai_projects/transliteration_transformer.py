from ast import literal_eval as ast_literal_eval
from time import time as get_time
from math import log as math_log
from re import compile as re_compile,sub as re_sub
from json import load as json_load
from argparse import Namespace
from collections import namedtuple
from urduhack import normalize as shahmukhi_normalize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import torch
from typing import List
import tempfile
import zipfile
from fairseq import checkpoint_utils, options, utils , tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


def pack_constraints(batch_constraints: List[List[torch.Tensor]]) -> torch.Tensor:
    max_constraints_len = 1
    for sentence_constraints in batch_constraints:
        if len(sentence_constraints):
            # number of constraints, plus sum of constrain lens, plus a zero after each
            constraints_len = (
                1
                + sum([c.size(0) for c in sentence_constraints])
                + len(sentence_constraints)
            )
            max_constraints_len = max(max_constraints_len, constraints_len)

    batch_size = len(batch_constraints)
    constraints_tensor = torch.zeros((batch_size, max_constraints_len)).long()
    for i, sentence_constraints in enumerate(batch_constraints):
        constraints_tensor[i, 0] = len(sentence_constraints)
        offset = 1
        for j, constraint in enumerate(sentence_constraints):
            this_len = constraint.size(0)
            constraints_tensor[i, offset : offset + this_len] = constraint
            offset += this_len + 1

    return constraints_tensor.long()


def unpack_constraints(constraint_tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Transforms *one row* of a packed constraint tensor (e.g., for one
    sentence in the batch) into a list of constraint tensors.
    """
    constraint_list = []
    num_constraints = constraint_tensor[0]
    constraints = constraint_tensor.tolist()
    offset = 1
    for i in range(num_constraints):
        where = constraints.index(0, offset)
        constraint_list.append(constraint_tensor[offset:where])
        offset = where + 1

    return constraint_list


from importlib.resources import files

RTL_LANG_CODES = {
    'ks',
    'pnb',
    'sd',
    'skr',
    'ur',
    'dv'
}

LANG_CODE_TO_SCRIPT_CODE = {
    "as"   : "Beng",
    "bn"   : "Beng",
    "doi"  : "Deva",
    "dv"   : "Thaa",
    "gom"  : "Deva",
    "gu"   : "Gujr",
    "hi"   : "Deva",
    "ks"   : "Aran",
    "mai"  : "Deva",
    "mr"   : "Deva",
    "ne"   : "Deva",
    "or"   : "Orya",
    "pa"   : "Guru",
    "pnb"  : "Aran",
    "sa"   : "Deva",
    "sd"   : "Arab",
    "si"   : "Sinh",
    "skr"  : "Aran",
    "ur"   : "Aran",
    "kn"   : "Knda",
    "ml"   : "Mlym",
    "ta"   : "Taml",
    "te"   : "Telu",
    "brx"  : "Deva",
    "mni"  : "Mtei",
    "sat"  : "Olck",
    "en"   : "Latn",
}

SCRIPT_CODE_TO_UNICODE_CHARS_RANGE_STR = {
    "Beng": "\u0980-\u09FF",
    "Deva": "\u0900-\u097F",
    "Gujr": "\u0A80-\u0AFF",
    "Guru": "\u0A00-\u0A7F",
    "Orya": "\u0B00-\u0B7F",
    "Knda": "\u0C80-\u0CFF",
    "Mlym": "\u0D00-\u0D7F",
    "Sinh": "\u0D80-\u0DFF",
    "Taml": "\u0B80-\u0BFF",
    "Telu": "\u0C00-\u0C7F",
    "Mtei": "\uABC0-\uABFF",
    "Arab": "\u0600-\u06FF\u0750-\u077F\u0870-\u089F\u08A0-\u08FF", 
    "Aran": "\u0600-\u06FF\u0750-\u077F\u0870-\u089F\u08A0-\u08FF", 
    "Latn": "\u0041-\u005A\u0061-\u007A", 
    "Olck": "\u1C50-\u1C7F",
    "Thaa": "\u0780-\u07BF",
}

INDIC_TO_LATIN_PUNCT = {
    '।': '.', 
    '॥': "..",  
    '෴': '.', 
    '꫰': ',',
    '꯫': '.',
    '᱾': '.',
    '᱿': '..',
    '۔': '.',
    '؟': '?',
    '،': ',',
    '؛': ';',
    '۝': "..",
}

INDIC_TO_LATIN_PUNCT_TRANSLATOR = str.maketrans(INDIC_TO_LATIN_PUNCT)

NON_LATIN_FULLSTOP_LANGS = {
    'as' : '।',
    'bn' : '।',
    'brx': '।',
    'doi': '।',
    'hi' : '।',
    'mai': '।',
    'mni': '꯫',
    'ne' : '।',
    'or' : '।',
    'pa' : '।',
    'sa' : '।',
    'sat': '᱾',
    'ks' : '۔',
    'pnb': '۔',
    'skr': '۔',
    'ur' : '۔',
}

ENDS_WITH_LATIN_FULLSTOP_REGEX = re_compile("(^|.*[^.])\.$")

def nativize_latin_fullstop(text, lang_code):
    if lang_code in NON_LATIN_FULLSTOP_LANGS and ENDS_WITH_LATIN_FULLSTOP_REGEX.match(text):
        return text[:-1] + NON_LATIN_FULLSTOP_LANGS[lang_code]
    return text

LATIN_TO_PERSOARABIC_PUNCTUATIONS = {
    '?': '؟',
    ',': '،',
    ';': '؛',
}

LATIN_TO_PERSOARABIC_PUNC_TRANSLATOR = str.maketrans(LATIN_TO_PERSOARABIC_PUNCTUATIONS)

SCRIPT_CODE_TO_NUMERALS = {
    "Beng": "০১২৩৪৫৬৭৮৯",
    "Deva": "०१२३४५६७८९",
    "Gujr": "૦૧૨૩૪૫૬૭૮૯",
    "Guru": "੦੧੨੩੪੫੬੭੮੯",
    "Orya": "୦୧୨୩୪୫୬୭୮୯",
    "Knda": "೦೧೨೩೪೫೬೭೮೯",
    "Mlym": "൦൧൨൩൪൫൬൭൮൯",
    "Sinh": "෦෧෨෩෪෫෬෭෮෯",
    "Taml": "௦௧௨௩௪௫௬௭௮௯",
    "Telu": "౦౧౨౩౪౫౬౭౮౯",
    "Mtei": "꯰꯱꯲꯳꯴꯵꯶꯷꯸꯹",
    "Arab": "۰۱۲۳۴۵۶۷۸۹", 
    "Aran": "۰۱۲۳۴۵۶۷۸۹", 
    "Latn": "0123456789",
    "Olck": "᱐᱑᱒᱓᱔᱕᱖᱗᱘᱙",
    "Thaa": "٠١٢٣٤٥٦٧٨٩", 
}

LANG_CODE_TO_NUMERALS = {
    lang_code: SCRIPT_CODE_TO_NUMERALS[script_code]
    for lang_code, script_code in LANG_CODE_TO_SCRIPT_CODE.items()
}

INDIC_TO_STANDARD_NUMERALS_GLOBAL_MAP = {}

for lang_code, lang_numerals in LANG_CODE_TO_NUMERALS.items():
    map_dict = {lang_numeral: en_numeral for lang_numeral, en_numeral in zip(lang_numerals, LANG_CODE_TO_NUMERALS["en"])}
    INDIC_TO_STANDARD_NUMERALS_GLOBAL_MAP.update(map_dict)

INDIC_TO_STANDARD_NUMERALS_TRANSLATOR = str.maketrans(INDIC_TO_STANDARD_NUMERALS_GLOBAL_MAP)

NATIVE_TO_LATIN_NUMERALS_TRANSLATORS = {
    lang_code: str.maketrans({lang_numeral: en_numeral for lang_numeral, en_numeral in zip(lang_numerals, LANG_CODE_TO_NUMERALS["en"])})
    for lang_code, lang_numerals in LANG_CODE_TO_NUMERALS.items()
    if lang_code != "en"
}

LATIN_TO_NATIVE_NUMERALS_TRANSLATORS = {
    lang_code: str.maketrans({en_numeral: lang_numeral for en_numeral, lang_numeral in zip(LANG_CODE_TO_NUMERALS["en"], lang_numerals)})
    for lang_code, lang_numerals in LANG_CODE_TO_NUMERALS.items()
    if lang_code != "en"
}

WORDFINAL_INDIC_VIRAMA_REGEX = re_compile("(\u09cd|\u094d|\u0acd|\u0a4d|\u0b4d|\u0ccd|\u0d4d|\u0dca|\u0bcd|\u0c4d|\uaaf6)$")

def hardfix_wordfinal_virama(word):

    return WORDFINAL_INDIC_VIRAMA_REGEX.sub("\\1\u200c", word)

ODIA_CONFUSING_YUKTAKSHARA_REGEX = re_compile("(\u0b4d)(ବ|ଵ|ୱ|ଯ|ୟ)")

def fix_odia_confusing_ambiguous_yuktakshara(word):

    return ODIA_CONFUSING_YUKTAKSHARA_REGEX.sub("\\1\u200c\\2", word)

LATIN_WORDFINAL_CONSONANTS_CHECKER_REGEX = re_compile(".*([bcdfghjklmnpqrstvwxyz])$")

DEVANAGARI_WORDFINAL_CONSONANTS_REGEX = re_compile("([\u0915-\u0939\u0958-\u095f\u0979-\u097c\u097e-\u097f])$")

def explicit_devanagari_wordfinal_schwa_delete(roman_word, indic_word):
    if LATIN_WORDFINAL_CONSONANTS_CHECKER_REGEX.match(roman_word):
        indic_word = DEVANAGARI_WORDFINAL_CONSONANTS_REGEX.sub("\\1\u094d", indic_word)
    return indic_word

def rreplace(text, find_pattern, replace_pattern, match_count=1):
    splits = text.rsplit(find_pattern, match_count)
    return replace_pattern.join(splits)

LANG_WORD_REGEXES = {
    lang_name: re_compile(f"[{SCRIPT_CODE_TO_UNICODE_CHARS_RANGE_STR[script_name]}]+")
    for lang_name, script_name in LANG_CODE_TO_SCRIPT_CODE.items()
}

normalizer_factory = IndicNormalizerFactory()





Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if cfg.generation.constraints:

        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    if cfg.generation.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )

class Transliterator:

    def __init__(
        self, data_bin_dir, model_checkpoint_path, lang_pairs_csv, lang_list_file, beam,device, batch_size = 32,
    ):
        self.parser = options.get_interactive_generation_parser()
        self.parser.set_defaults(
            path = model_checkpoint_path,
            num_wokers = -1,
            batch_size = batch_size,
            buffer_size = batch_size + 1,
            task = "translation_multi_simple_epoch",
            beam = beam,

        )

        self.args = options.parse_args_and_arch(self.parser, input_args = [data_bin_dir] )

        self.args.skip_invalid_size_inputs_valid_test = False

        self.args.lang_pairs = lang_pairs_csv

        self.args.lang_dict = lang_list_file

        self.cfg = convert_namespace_to_omegaconf(self.args)

        if isinstance(self.cfg, Namespace):
            self.cfg = convert_namespace_to_omegaconf(self.cfg)

        self.total_translate_time = 0

        utils.import_user_module(self.cfg.common)

        if self.cfg.interactive.buffer_size < 1:
            self.cfg.interactive.buffer_size = 1
        if self.cfg.dataset.max_tokens is None and self.cfg.dataset.batch_size is None:
            self.cfg.dataset.batch_size = 1

        assert (
            not self.cfg.generation.sampling or self.cfg.generation.nbest == self.cfg.generation.beam
        ), "--sampling requires --nbest to be equal to --beam"
        assert (
            not self.cfg.dataset.batch_size
            or self.cfg.dataset.batch_size <= self.cfg.interactive.buffer_size
        ), "--batch-size cannot be larger than --buffer-size"

        self.use_cuda = device.type == "cuda"

        self.task = tasks.setup_task(self.cfg.task)

        overrides = ast_literal_eval(self.cfg.common_eval.model_overrides)

        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.cfg.common_eval.path),
            arg_overrides=overrides,
            task=self.task,
            suffix=self.cfg.checkpoint.checkpoint_suffix,
            strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=self.cfg.checkpoint.checkpoint_shard_count,
        )

        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        for i in range(len(self.models)):
            if self.models[i] is None:
                continue
            if self.cfg.common.fp16:
                self.models[i].half()

            if self.use_cuda and not self.cfg.distributed_training.pipeline_model_parallel:
                self.models[i].cuda()
            self.models[i].prepare_for_inference_(self.cfg)

        self.generator = self.task.build_generator(self.models, self.cfg.generation)

        self.tokenizer = self.task.build_tokenizer(self.cfg.tokenizer)
        self.bpe = self.task.build_bpe(self.cfg.bpe)

        self.align_dict = utils.load_align_dict(self.cfg.generation.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x

    def translate(self, inputs, nbest=1):

        start_id = 0

        results = []
        for batch in make_batches(inputs, self.cfg, self.task, self.max_positions, self.encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }

            translate_start_time = get_time()
            translations = self.task.inference_step(
                self.generator, self.models, sample, constraints=constraints
            )
            translate_time = get_time() - translate_start_time
            self.total_translate_time += translate_time
            list_constraints = [[] for _ in range(bsz)]
            if self.cfg.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                constraints = list_constraints[i]
                results.append(
                    (
                        start_id + id,
                        src_tokens_i,
                        hypos,
                        {
                            "constraints": constraints,
                            "time": translate_time / len(translations),
                        },
                    )
                )

        result_str = ""
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):

            src_str = ""
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.cfg.common_eval.post_process)

                result_str += "S-{}\t{}".format(id_, src_str) + '\n'

                result_str += "W-{}\t{:.3f}\tseconds".format(id_, info["time"]) + '\n'

                for constraint in info["constraints"]:

                    result_str += "C-{}\t{}".format(
                            id_,
                            self.tgt_dict.string(constraint, self.cfg.common_eval.post_process),
                        ) + '\n'

            for hypo in hypos[: min(len(hypos), nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                )
                detok_hypo_str = self.decode_fn(hypo_str)
                score = hypo["score"] / math_log(2)  

                result_str += "H-{}\t{}\t{}".format(id_, score, hypo_str) + '\n'

                result_str += "D-{}\t{}\t{}".format(id_, score, detok_hypo_str) + '\n'

                result_str += "P-{}\t{}".format(
                        id_,
                        " ".join(
                            map(
                                lambda x: "{:.4f}".format(x),

                                hypo["positional_scores"].div_(math_log(2)).tolist(),
                            )
                        ),
                    ) + '\n'

                if self.cfg.generation.print_alignment:
                    alignment_str = " ".join(
                        ["{}-{}".format(src, tgt) for src, tgt in alignment]
                    )

                    result_str += "A-{}\t{}".format(id_, alignment_str) + '\n'

        return result_str


class Transliteration_Transformer:
    def __init__(self, word_prob_dicts_files,model_file,corpus_zip,tgt_langs, device, beam_width=4, rescore=True):
        self.all_supported_langs = {'as', 'bn', 'brx', 'gom', 'gu', 'hi', 'kn', 'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sd', 'si', 'ta', 'te', 'ur'}
        lang_list_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        lang_list_file.write("en\n"+"\n".join(self.all_supported_langs))
        lang_list_file.close()
        corpus_dir = tempfile.mkdtemp()
        zipfile.ZipFile(corpus_zip, 'r').extractall(corpus_dir)
        if 'en' in tgt_langs:
            lang_pairs = [lang+"-en" for lang in self.all_supported_langs]
        else:
            lang_pairs = ["en-"+lang for lang in self.all_supported_langs]
        with torch.serialization.safe_globals([Namespace]):
            self.transliterator = Transliterator(
                corpus_dir,
                model_file,
                lang_pairs_csv = ','.join(lang_pairs),
                lang_list_file=lang_list_file.name,
                device=device,
                beam = beam_width, 
                batch_size = 32,
            )
        self.beam_width = beam_width
        self._rescore = rescore
        if self._rescore:
            self.word_prob_dicts={lang:json_load(open(word_prob_dicts_files[lang])) for lang in tgt_langs}

    def indic_normalize(self, words, lang_code):
        if lang_code not in ['gom', 'ks', 'ur', 'mai', 'brx', 'mni']:
            normalizer = normalizer_factory.get_normalizer(lang_code)
            words = [ normalizer.normalize(word) for word in words ]

        if lang_code in ['mai', 'brx' ]:
            normalizer = normalizer_factory.get_normalizer('hi')
            words = [ normalizer.normalize(word) for word in words ]

        if lang_code in [ 'ur' ]:
            words = [ shahmukhi_normalize(word) for word in words ]

        if lang_code == 'gom':
            normalizer = normalizer_factory.get_normalizer('kK')
            words = [ normalizer.normalize(word) for word in words ]

        return words

    def pre_process(self, words, src_lang, tgt_lang):

        if src_lang != 'en':
            self.indic_normalize(words, src_lang)

        words = [' '.join(list(word.lower())) for word in words]

        lang_code = tgt_lang if src_lang == 'en' else src_lang

        words = ['__'+ lang_code +'__ ' + word for word in words]

        return words

    def rescore(self, res_dict, result_dict, tgt_lang, alpha ):

        alpha = alpha

        word_prob_dict = self.word_prob_dicts[tgt_lang]

        candidate_word_prob_norm_dict = {}
        candidate_word_result_norm_dict = {}

        input_data = {}
        for i in res_dict.keys():
            input_data[res_dict[i]['S']] = []
            for j in range(len(res_dict[i]['H'])):
                input_data[res_dict[i]['S']].append( res_dict[i]['H'][j][0] )

        for src_word in input_data.keys():
            candidates = input_data[src_word]

            candidates = [' '.join(word.split(' ')) for word in candidates]

            total_score = 0

            if src_word.lower() in result_dict.keys():
                for candidate_word in candidates:
                    if candidate_word in result_dict[src_word.lower()].keys():
                        total_score += result_dict[src_word.lower()][candidate_word]

            candidate_word_result_norm_dict[src_word.lower()] = {}

            for candidate_word in candidates:
                candidate_word_result_norm_dict[src_word.lower()][candidate_word] = (result_dict[src_word.lower()][candidate_word]/total_score)

            candidates = [''.join(word.split(' ')) for word in candidates ]

            total_prob = 0 

            for candidate_word in candidates:
                if candidate_word in word_prob_dict.keys():
                    total_prob += word_prob_dict[candidate_word]        

            candidate_word_prob_norm_dict[src_word.lower()] = {}
            for candidate_word in candidates:
                if candidate_word in word_prob_dict.keys():
                    candidate_word_prob_norm_dict[src_word.lower()][candidate_word] = (word_prob_dict[candidate_word]/total_prob)

        output_data = {}
        for src_word in input_data.keys():

            temp_candidates_tuple_list = []
            candidates = input_data[src_word]
            candidates = [ ''.join(word.split(' ')) for word in candidates]

            for candidate_word in candidates:
                if candidate_word in word_prob_dict.keys():
                    temp_candidates_tuple_list.append((candidate_word, alpha*candidate_word_result_norm_dict[src_word.lower()][' '.join(list(candidate_word))] + (1-alpha)*candidate_word_prob_norm_dict[src_word.lower()][candidate_word] ))
                else:
                    temp_candidates_tuple_list.append((candidate_word, 0 ))

            temp_candidates_tuple_list.sort(key = lambda x: x[1], reverse = True )

            temp_candidates_list = []
            for cadidate_tuple in temp_candidates_tuple_list: 
                temp_candidates_list.append(' '.join(list(cadidate_tuple[0])))

            output_data[src_word] = temp_candidates_list

        return output_data

    def post_process(self, translation_str, tgt_lang):
        lines = translation_str.split('\n')

        list_s = [line for line in lines if 'S-' in line]

        list_h = [line for line in lines if 'H-' in line]

        list_s.sort(key = lambda x: int(x.split('\t')[0].split('-')[1]) )

        list_h.sort(key = lambda x: int(x.split('\t')[0].split('-')[1]) )

        res_dict = {}
        for s in list_s:
            s_id = int(s.split('\t')[0].split('-')[1])

            res_dict[s_id] = { 'S' : s.split('\t')[1] }

            res_dict[s_id]['H'] = []

            for h in list_h:
                h_id = int(h.split('\t')[0].split('-')[1])

                if s_id == h_id:
                    res_dict[s_id]['H'].append( ( h.split('\t')[2], pow(2,float(h.split('\t')[1])) ) )

        for r in res_dict.keys():
            res_dict[r]['H'].sort(key = lambda x : float(x[1]) ,reverse =True)

        result_dict = {}
        for i in res_dict.keys():            
            result_dict[res_dict[i]['S']] = {}
            for j in range(len(res_dict[i]['H'])):
                 result_dict[res_dict[i]['S']][res_dict[i]['H'][j][0]] = res_dict[i]['H'][j][1]

        transliterated_word_list = []
        if self._rescore:
            output_dir = self.rescore(res_dict, result_dict, tgt_lang, alpha = 0.9)            
            for src_word in output_dir.keys():
                for j in range(len(output_dir[src_word])):
                    transliterated_word_list.append( output_dir[src_word][j] )

        else:
            for i in res_dict.keys():

                for j in range(len(res_dict[i]['H'])):
                    transliterated_word_list.append( res_dict[i]['H'][j][0] )

        transliterated_word_list = [''.join(word.split(' ')) for word in transliterated_word_list]

        return transliterated_word_list

    def _transliterate_word(self, text, src_lang, tgt_lang, topk=4, nativize_punctuations=True, nativize_numerals=False):
        if not text:
            return text
        text = text.lower().strip()

        if src_lang != 'en':

            text = text.translate(INDIC_TO_LATIN_PUNCT_TRANSLATOR)
            text = text.translate(INDIC_TO_STANDARD_NUMERALS_TRANSLATOR)
        else:

            if nativize_punctuations:
                if tgt_lang in RTL_LANG_CODES:
                    text = text.translate(LATIN_TO_PERSOARABIC_PUNC_TRANSLATOR)
                text = nativize_latin_fullstop(text, tgt_lang)
            if nativize_numerals:
                text = text.translate(LATIN_TO_NATIVE_NUMERALS_TRANSLATORS[tgt_lang])

        matches = LANG_WORD_REGEXES[src_lang].findall(text)

        if not matches:
            return [text]

        src_word = matches[-1]

        transliteration_list = self.batch_transliterate_words([src_word], src_lang, tgt_lang, topk=topk)[0]

        if tgt_lang != 'en' or tgt_lang != 'sa':

            for i in range(len(transliteration_list)):
                transliteration_list[i] = hardfix_wordfinal_virama(transliteration_list[i])

        if src_word == text:
            return transliteration_list

        return [
            rreplace(text, src_word, tgt_word)
            for tgt_word in transliteration_list
        ]

    def batch_transliterate_words(self, words, src_lang, tgt_lang, topk=4):
        perprcossed_words = self.pre_process(words, src_lang, tgt_lang)
        translation_str = self.transliterator.translate(perprcossed_words, nbest=topk)

        transliteration_list = self.post_process(translation_str, tgt_lang)

        if tgt_lang == 'mr':
            for i in range(len(transliteration_list)):
                transliteration_list[i] = transliteration_list[i].replace("अॅ", 'ॲ')

        if tgt_lang == 'or':
            for i in range(len(transliteration_list)):
                transliteration_list[i] = fix_odia_confusing_ambiguous_yuktakshara(transliteration_list[i])

        if tgt_lang == 'sa':
            for i in range(len(transliteration_list)):
                transliteration_list[i] = explicit_devanagari_wordfinal_schwa_delete(words[0], transliteration_list[i])

            transliteration_list = list(dict.fromkeys(transliteration_list))

        return [transliteration_list]

    def _transliterate_sentence(self, text, src_lang, tgt_lang, nativize_punctuations=True, nativize_numerals=False):

        if not text:
            return text
        text = text.lower().strip()

        if src_lang != 'en':

            text = text.translate(INDIC_TO_LATIN_PUNCT_TRANSLATOR)
            text = text.translate(INDIC_TO_STANDARD_NUMERALS_TRANSLATOR)
        else:

            if nativize_punctuations:
                if tgt_lang in RTL_LANG_CODES:
                    text = text.translate(LATIN_TO_PERSOARABIC_PUNC_TRANSLATOR)
                text = nativize_latin_fullstop(text, tgt_lang)
            if nativize_numerals:
                text = text.translate(LATIN_TO_NATIVE_NUMERALS_TRANSLATORS[tgt_lang])

        matches = LANG_WORD_REGEXES[src_lang].findall(text)

        if not matches:
            return text

        out_str = text
        for match in matches:
            result = self.batch_transliterate_words([match], src_lang, tgt_lang)[0][0]
            out_str = re_sub(match, result, out_str, 1)
        return out_str
