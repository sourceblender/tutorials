# train.py
"""Train three models on SST-2's train split (the eval uses the validation split).

incumbent: what we run today   - single words, 2,000 examples
candidate: the proposed change - words and word pairs, 20,000 examples
broken:    the candidate recipe trained on SHUFFLED labels - a known-bad change
           the eval must reject, or the eval can't detect anything
"""
import random
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

REVISION = "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"  # same pinned commit as freeze.py
train = load_dataset("stanfordnlp/sst2", split="train", revision=REVISION).shuffle(seed=7)

def fit(n, ngrams, shuffle_labels=False):
    rows = train.select(range(n))
    x, y = rows["sentence"], list(rows["label"])
    if shuffle_labels:
        random.Random(7).shuffle(y)
    model = make_pipeline(TfidfVectorizer(ngram_range=ngrams, min_df=2),
                          LogisticRegression(max_iter=1000))
    return model.fit(x, y)

joblib.dump(fit(2_000, (1, 1)), "incumbent.joblib")
joblib.dump(fit(20_000, (1, 2)), "candidate.joblib")
joblib.dump(fit(20_000, (1, 2), shuffle_labels=True), "broken.joblib")
print("trained: incumbent, candidate, broken")
