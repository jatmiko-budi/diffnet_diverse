from util.tokenization.id.crf_tokenizer import CRFTokenizer

from util.language_detection.crf_language_predictor import CRFLanguagePredictor
from util.language_detection.cnn_language_predictor import CNNLanguagePredictor
from util.language_detection.rnn_language_predictor import RNNLanguagePredictor

from id_preprocessing.normalization_rule import NormalizationRule
from id_preprocessing.formalization import TextFormalizer
from util.smt.smt_wrapper import SMTWrapper
from util.google_translete_client import GoogleTranslateClient

from collections import defaultdict

class CodeSwitchNormalization():
    def __init__(self, language_detection="crf", google_api_key=None, smt_unit="word"):
        """
        :param language_detection   : string, language algorithm to be used, valid values are
                                        "crf"/"cnn"/"rnn"  
        :param google_api_key       : string, valid api key for goole translation api 
        :param smt_unit             : string, unit translation for slang translator, valid values are
                                        "char"/"word"
        """
        if language_detection == "crf":
            self.language_detector = CRFLanguagePredictor()
        elif language_detection == "cnn":
            self.language_detector = CNNLanguagePredictor()
        elif language_detection == "rnn":
            self.language_detector = RNNLanguagePredictor()
        self.language_detector.load_model()
    
        self.tokenizer = CRFTokenizer()
        self.tokenizer.load_model()

        self.slang_translator = SMTWrapper(unit_translation=smt_unit)
        self.slang_translator.load_model()

        self.rule_normalization = NormalizationRule()
        self.lexical_normalization = TextFormalizer()

        self.tranlation_client = {
            ("i", "s"):GoogleTranslateClient("id", "su", google_api_key),
            ("s", "i"):GoogleTranslateClient("su", "id", google_api_key),
            ("i", "j"):GoogleTranslateClient("id", "jv", google_api_key),
            ("j", "i"):GoogleTranslateClient("jv", "id", google_api_key)
        }

    def translate(self, text_list, skip_normalization=False):
        """
        :param text_list            : either list of string of list of token list, 
                                        if list of string is given, CRF tokenizer will be used,
                                        if list of token list is given, tokenization will be skipped
        :param skip_normalization   : boolean, either to skip slang normalization step or not
        :return result              : list of string containing code-switch normalization result
        """
        if isinstance(text_list[0], str):
            text_list = self.tokenizer.tokenize(text_list)

        if not skip_normalization:
            text_list = self.normalize(text_list)
        lang_sequence_list = self.language_detector.test(text_list)
        result = []
        for text, lang_sequence in zip(text_list, lang_sequence_list):
            matrix_lang, embed_lang = self.get_matrix_embed_language(lang_sequence)
            translated_text = " ".join(text)
            if embed_lang != "":
                translated_text = self.tranlation_client[(embed_lang, matrix_lang)].translate(translated_text)
            if matrix_lang != "i":
                translated_text = self.tranlation_client[(matrix_lang, "i")].translate(translated_text)
            result.append(translated_text)
        return result

    def normalize(self, text_list):
        """
        :param text_list: list of token list, normalization inputs
        :return result  : list of token list, normalization results
        """
        result = []
        for text in text_list:
            result.append(
                self.rule_normalization.run(
                    self.lexical_normalization.formalize_from_dictionary(text)
                )
            )
        return self.slang_translator.translate(result)
    
    def get_matrix_embed_language(self, lang_sequence):
        """
        :param lang_sequence                        : list of language detection label
        :return matrix_lang_code, embed_lang_code   : string, language code for matrix language and embedding language  
        """
        lang_counter = defaultdict(int)
        for lang in lang_sequence:
            lang_counter[lang] += 1
        lang_list = [k for k, v in sorted(lang_counter.items(), key=lambda item: item[1])]
        embed_lang_code = ""
        matrix_lang_code = ""
        for lang in lang_list:
            if lang != "o" and matrix_lang_code == "":
                matrix_lang_code = lang
            elif lang != "o" and matrix_lang_code != "":
                embed_lang_code = lang
                break
        return matrix_lang_code, embed_lang_code