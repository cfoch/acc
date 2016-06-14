import enchant
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
from settings import ALLOWED_WORDS_PATH
from utils import ne_tree_to_list


class ACCTweetTokenizer(TweetTokenizer):
    DEFAULT_SYMBOLS = "!@#$%*()_+{}:>?«»\"\'.,-"
    DEFAULT_NEGATION_WORDS = ["not", "can't", "isn't", "shouldn't", "doesn't",
                              "don't"]

    def __init__(self, collocations=True, remove_symbols=True,
                 check_word_correctness=True, stemming=True,
                 join_negation_words=True, symbols_list=None,
                 negation_words_list=None):

        TweetTokenizer.__init__(self)
        self.collocations = collocations
        self.remove_symbols = remove_symbols
        self.check_word_correctness = check_word_correctness
        self.stemming = stemming
        self.join_negation_words = join_negation_words

        if symbols_list is None:
            self.symbols_list = self.DEFAULT_SYMBOLS
        else:
            self.symbols_list = symbols_list

        if negation_words_list is None:
            self.negation_words_list = self.DEFAULT_NEGATION_WORDS
        else:
            self.negation_words_list = negation_words_list

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
        if self.join_negation_words:
            tokens = self._join_negation_words(tokens)
        if self.stemming:
            stemmer = SnowballStemmer('english')
            tokens = [stemmer.stem(token) for token in tokens]
        return tokens

    def _filter_dictionary(self, tokens):
        d = enchant.DictWithPWL("en_US", ALLOWED_WORDS_PATH)
        dummy = []
        for token in tokens:
            if ((len(token.split(" ")) == 1 and d.check(token)) or
                    len(token.split(" ")) > 1):
                dummy.append(token)
        return dummy

    def _join_negation_words(self, tokens):
        dummy = []
        just_joined_words = False
        for i in range(len(tokens)):
            if just_joined_words:
                just_joined_words = False
                continue
            if tokens[i] in self.negation_words_list and i + 1 < len(tokens):
                token = tokens[i] + " " + tokens[i + 1]
                just_joined_words = True
            else:
                token = tokens[i]
            dummy.append(token)
        return dummy
