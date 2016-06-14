from IPython import embed
import numpy
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer,\
    _document_frequency
from data import get_negative_words, get_positive_words


def idf_deformer(vocabulary, factor):
    deformer = numpy.full(len(vocabulary), fill_value=1)
    positive_words, negative_words = get_positive_words(), get_negative_words()

    for i, token in enumerate(vocabulary):
        words = token.split(" ")
        if len(set(words).intersection(set(positive_words))) > 0:
            deformer[i] += factor
        elif len(set(words).intersection(set(negative_words))) > 0:
            deformer[i] -= factor
    return deformer
    

class ACCTransformer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        TfidfTransformer.__init__(self, *args, **kwargs)

    def fit(self, X, y=None, idf_deformer=None):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = numpy.log(float(n_samples) / df) + 1.0
            if idf_deformer is not None:
                idf *= idf_deformer
            self._idf_diag = sp.spdiags(idf,
                                        diags=0, m=n_features, n=n_features)
        return self


class ACCVectorizer(TfidfVectorizer):
    def __init__(self, deform_factor=0.75, *args, **kwargs):
        TfidfVectorizer.__init__(self, *args, **kwargs)
        self._tfidf = ACCTransformer(norm=self.norm, use_idf=self.use_idf,
                                       smooth_idf=self.smooth_idf,
                                       sublinear_tf=self.sublinear_tf)
        self.deform_factor = deform_factor

    def fit_transform(self, raw_documents, y=None):
        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        vocabulary = self.get_feature_names()
        deformer = idf_deformer(vocabulary, self.deform_factor)
        self._tfidf.fit(X, idf_deformer=deformer)
        return self._tfidf.transform(X, copy=False)
