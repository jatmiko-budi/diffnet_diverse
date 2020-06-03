import re
from language_engine.config import LibPathConfig
from language_engine.util.universal_util import repetition_reduction

class NormalizationRule():
    # known Error case:
    # ny-   : counter ex -> nyata (formal word)
    # ng-   : counter ex -> nggak (slang in different form)
    # ng-   : may be replace by me-
    # -x & -q : often used as replacement of letter 'k'
    # "     : may be a case of improper whitespacing
    # -in   : may also be replaced by -i
    # reduplication : lexical analysis of the subtoken before "2" may be needed
    #                   ex: berkali2 -> berkali-kali not berkali-berkali

    # known possible rule but not included due to its inconsistencies
    # 'a: nya
    # multi token duplication; ["dua", "duanya"] -> ["dua-duanya"]

    duplication_marker = ["2", "\""]
    prefix_slang = {"ng":"meng", "ny":"meny", "ga":"tidak "}
    suffix_slang = {
                    "x":"nya", "ny":"nya","xa":"nya", "\'y":"nya",
                    "q":"ku", "in":"kan"
                    }
    numeric_slang = {
                        "plh":" puluh", "rb":" ribu", "k":" ribu", "jt":" juta",
                        "hr":" hari", "bln":" bulan",
                        "x":" kali"
                    }

    def __init__(self, custom_ruleset={}):
        """
        :params custom_ruleset: dictionary of slang, the key is regex/regular string, the value is a replacement string 
        """
        self.config = LibPathConfig()
        self.custom_ruleset = custom_ruleset
        self.vocab = set()
        with open(self.config.get_vocab_list_path(), "r") as f:
            for line in f:
                self.vocab.add(line.rstrip())

    def run(self, tokens):
        """
        :params tokens: list of string that will be normalized
        :return tokens: list of normalized string
        """
        for idx, token in enumerate(tokens):
            if token in self.vocab:
                tokens[idx] = token
                continue
            token = self.prefix_normalization(token)
            token = self.numeric_normalization(token)
            token = self.suffix_normalization(token)
            token = self.reduplication(token)
            token = repetition_reduction(token, 1)
            token = repetition_reduction(token, 2)
            tokens[idx] = self.custom_normalization(token)
        return tokens

    def prefix_normalization(self, token):
        """
        :params token: string that its prefix will be normalized by default ruleset
        :return token: normalized string
        """
        for slang, sub in self.prefix_slang.items():
            pattern = r"^"+re.escape(slang)
            token = re.sub(pattern, sub, token)
        return token
            
    def suffix_normalization(self, token):
        """
        :params token: string that its suffix will be normalized by default ruleset
        :return token: normalized string
        """
        for slang, sub in self.suffix_slang.items():
            pattern = re.escape(slang) + r"$"
            token = re.sub(pattern, sub, token)
        return token

    def numeric_normalization(self, token):
        """
        :params token: string that prefixed by numeric char, 
                        its suffix will be normalized by default ruleset
        :return token: normalized string
        """
        for slang, sub in self.numeric_slang.items():
            pattern = r"^\d+(" + re.escape(slang) + r")$"
            pattern_match = re.match(pattern, token)
            try:
                detected_slang = pattern_match.group(1)
                token = re.sub(detected_slang, sub, token)
            except AttributeError:
                pass
        return token

    def custom_normalization(self, token):
        """
        :params token: string that will be normalized by user's ruleset
        :return token: normalized string
        """
        for slang, sub in self.custom_ruleset.items():
            token = re.sub(slang, sub, token)
        return token

    def reduplication(self, token):
        """
        :params token: string with one of duplication marker (2/") in the middle or end, 
                        numeric only token will be skipped 
        :return token: normalized string
        """
        # skip numeric only token
        if re.search("^\d+$", token):
            return token
        for marker in self.duplication_marker:
            subwords = token.split(marker)
            if len(subwords) > 1:
                token = "-".join([subwords[0]]*2)
                if len(subwords) > 1:
                    token += subwords[1] 
                return token
        else:
            return token
