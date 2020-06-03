import yake


class YakeKeywordExtractor:

    def __init__(self, max_ngram, language, number_of_keywords,
                 deduplication_thresold=0.8, window_size=2, deduplication_algo="seqm"):
        """

        :param max_ngram: maximum word in a phrase
        :param language: language code: e.g id for indonesian and en for english
        :param number_of_keywords: the number of term that will be generated as keywords
        :param deduplication_threshold: a number between 0-1
        :param window_size: a number represent sliding window width to generate term (phrase) candidate
        :param deduplication_algo: algorithm to remove duplicate term candidates
        """

        self.yake = yake.KeywordExtractor(lan=language, n=max_ngram, dedupLim=deduplication_thresold,
                                          dedupFunc=deduplication_algo,
                                          windowsSize=window_size,
                                          top=number_of_keywords,
                                          features=None)

    def extract_keywords(self, text):
        """
        :param text: a document (text) as in put to extract keywords from
        :return: a list of tuple (keyword, score) the lower the score, the more important the keyword is
        """
        return self.yake.extract_keywords(text)
