from nltk.translate import AlignedSent, IBMModel1, IBMModel2, IBMModel3, StackDecoder, PhraseTable
from nltk import lm
from math import log
from collections import defaultdict
import re, string
from language_engine.util.universal_util import is_emoji_only, is_punctuation_only, replace_user_tag, replace_hash_tag, replace_link

# TO DO:

# Handle UNK
# Pruning Vocab (emoji, punctuation, url,)
class SMTSlang():
    align_models = {
        "ibm1":IBMModel1,
        "ibm2":IBMModel2,
        "ibm3":IBMModel3
    }

    def __init__(self, model_no, n_gram):
        """
        :params model_no: string, align_model to be used, choice ['ibm1', 'ibm2', 'ibm3'] 
        :params n_gram: N for ngram languange model
        """
        self.ngram = n_gram
        self.align_model = self.align_models[model_no]
        self.lang_model = lm.MLE(self.ngram)
        self.source_vocab = set()
        self.target_vocab = set()

    def train(self, pair_sents, n_iter=10):
        """
        :params pair_sents: corpus in form of list of pair sentences [(source, target)]
        :params n_iter: align_model training iterations
        """
        # To do :
        # enable use of external vocab
        # save vocabulary on phrase table 
        train_pair = []
        for source_sents, target_sents in pair_sents:
            train_data = AlignedSent(self._filter_tokens(source_sents)[0], self._filter_tokens(target_sents)[0])
            train_pair.append(train_data)

        self.align_model = self.align_model(train_pair, n_iter)
        translation_table = PhraseTable()
        for source_token, target_list in self.align_model.translation_table.items():
            limit = 1000 if len(target_list.items()) > 1000 else len(target_list.items())
            counter = 0
            for target_token, probability in sorted(target_list.items(), key=lambda item: log(item[1])):
                if counter > limit:
                    break
                translation_table.add((source_token, ), (target_token, ), log(probability))
                counter += 1

        target_corpus = [target_sent for source_sent, target_sent in pair_sents]
        target_ngram, vocabulary = lm.preprocessing.padded_everygram_pipeline(self.ngram, target_corpus)
        self.lang_model.fit(target_ngram, vocabulary)
        language_prob = defaultdict(lambda: float("-inf"))
        for sent in target_ngram:
            for ngram in sent:
                language_prob[ngram] = self.lang_model.logscore(ngram[-1], ngram[:-1])
        language_model = type('',(object,),{'probability_change': lambda self, context, phrase: language_prob[phrase], 'probability': lambda self, phrase: language_prob[phrase]})() 
        self.decoder = StackDecoder(translation_table, language_model)

    def test(self, sents):
        """
        :params sents: tokenized sentence in form of list of string
        :return list: list of string, normalization result
        """
        # Todo :
        # revert special token
        # handle OOV token
        sents, emojis, puncts = self._filter_tokens(sents)
        for i, token in enumerate(sents):
            token,_ = replace_hash_tag(token)
            token,_ = replace_link(token)
            sents[i], _ = replace_user_tag(token)
        res = self.decoder.translate(sents)
        return res

    def _filter_tokens(self, tokens):
        """
        :params sents: tokenized sentence in form of list of string
        :return tokens: list of string, filtered_token
        :return emoji_tokens: list of detected and replaced emoji token
        :return punct_tokens: list of detected and replaced punctuation token
        """
        punct_tokens = []
        emoji_tokens = []
        for i, token in enumerate(tokens):
            if is_punctuation_only(token):
                punct_tokens.append(token)
                tokens[i] = "[PUNCT]"
                continue
            if is_emoji_only(token):
                emoji_tokens.append(token)
                tokens[i] = "[EMOJI]"
                continue
        return tokens, emoji_tokens, punct_tokens