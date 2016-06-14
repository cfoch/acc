import enchant
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer

from utils import ne_tree_to_list
from data import get_negative_contractions


class ACCTweetTokenizer(TweetTokenizer):
    DEFAULT_SYMBOLS = "!@#$%*()_+{}:>?«»\"\'.,-"
    DEFAULT_NEGATION_WORDS = ["not"]

    def __init__(self, collocations=True, remove_symbols=True,
                 check_word_correctness=True, stemming=True,
                 join_negative_contractions=True, symbols_list=None,
                 negative_contractions_list=None):

        TweetTokenizer.__init__(self)
        self.collocations = collocations
        self.remove_symbols = remove_symbols
        self.check_word_correctness = check_word_correctness
        self.stemming = stemming
        self.join_negative_contractions = join_negative_contractions

        if symbols_list is None:
            self.symbols_list = self.DEFAULT_SYMBOLS
        else:
            self.symbols_list = symbols_list

        if negative_contractions_list is None:
            self.negative_contractions_list = self.DEFAULT_NEGATION_WORDS
        else:
            self.negative_contractions_list = negative_contractions_list

    def tokenize(self, text):
        tokens = super(ACCTweetTokenizer, self).tokenize(text)
        if self.collocations:
            chunked = ne_chunk(pos_tag(tokens), binary=True)
            tokens = ne_tree_to_list(chunked)
        if self.remove_symbols:
            tokens = [token for token in tokens
                      if token not in self.symbols_list]
        if self.check_word_correctness:
            tokens = self._filter_dictionary(tokens)
        if self.join_negative_contractions:
            tokens = self._join_negative_contractions(tokens)
        if self.stemming:
            stemmer = SnowballStemmer('english')
            tokens = [stemmer.stem(token) for token in tokens]
        return tokens

    def _filter_dictionary(self, tokens):
        try:
            from settings import ALLOWED_WORDS_PATH
            d = enchant.DictWithPWL("en_US", ALLOWED_WORDS_PATH)
        except:
            d = enchant.Dict("en_US")
        dummy = []
        for token in tokens:
            if ((len(token.split(" ")) == 1 and d.check(token)) or
                    len(token.split(" ")) > 1):
                dummy.append(token)
        return dummy

    def _join_negative_contractions(self, tokens):
        dummy = []
        just_joined_words = False
        for i in range(len(tokens)):
            if just_joined_words:
                just_joined_words = False
                continue
            if (tokens[i] in get_negative_contractions() and
                    i + 1 < len(tokens)):
                token = tokens[i] + " " + tokens[i + 1]
                just_joined_words = True
            else:
                token = tokens[i]
            dummy.append(token)
        return dummy
