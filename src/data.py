import os
import settings

def get_negative_words():
    try:
        from settings import NEGATIVE_WORDS_PATH
        with open(NEGATIVE_WORDS_PATH) as f:
            return [w.strip() for w in f.readlines()]
    except:
        return []


def get_positive_words():
    try:
        from settings import POSITIVE_WORDS_PATH
        with open(POSITIVE_WORDS_PATH) as f:
            return [w.strip() for w in f.readlines()]
    except:
        return []


def get_negative_contractions():
    try:
        from settings import NEGATIVE_CONTRACTIONS_WORDS_PATH
        with open(NEGATIVE_CONTRACTIONS_WORDS_PATH) as f:
            return [w.strip() for w in f.readlines()]
    except:
        return []
