import pandas as pd
from language_engine.config import LibPathConfig


class TextFormalizer:

    def __init__(self, formalization_dict=None):
        """

        :param formalization_dict: Dictionary: a dictionary contains informal word as a key and it's formal
        ord as value. Default dictionary will be infered from this library resources
        """
        lib_conf = LibPathConfig()
        if formalization_dict is not None:
            self.formalization_dict = formalization_dict
        else:
            path = lib_conf.get_formalized_dict_path()
            formal_words = pd.read_csv(path, sep="\t")
            formal_dict = pd.Series(formal_words.formalized_word.values, index=formal_words.word).to_dict()
            self.formalization_dict = formal_dict

    def formalize_from_dictionary(self, tokenized_document):
        """

        :param tokenized_document: list: a list that contains token (word/terms) that has been tokenized and preprocessed
        :return: list: a list of mapped tokens (into formal word)

        Examples: input: ['Sy', 'mau', 'makan', 'tapi', 'sdg', 'g', 'lapar']
                  output: ['Saya', 'mau', 'makan', 'tapi', 'sedang', 'tidak', 'lapar']'
        """
        return list(map(lambda x: self.formalization_dict[x] if x in self.formalization_dict else x, tokenized_document))
