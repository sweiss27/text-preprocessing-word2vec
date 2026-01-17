# Text Data Preprocessing and Word Embedding in Python
# Setup

import sys
print("Python executable:", sys.executable)

# Install requirements (run in terminal, not in Python):
# pip install nltk pandas matplotlib seaborn gensim scikit-learn wordcloud


# Install requirements (run in terminal, not in Python):
# pip install nltk pandas matplotlib seaborn gensim scikit-learn wordcloud


import sys
print(sys.executable)

# Install requirements (run in terminal, not in Python):
# pip install nltk pandas matplotlib seaborn gensim scikit-learn wordcloud


# ## 2. Dataset
# **Using the NLTK Gutenberg corpus. File used: `austen-emma.txt`.**

 
import nltk
nltk.download('gutenberg', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("NLTK resources ready.")


import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

 
# ## 3. Preprocessing 
# Steps:
# - Cleaning: remove punctuation/numbers and extra whitespace
# - Normalization: lowercase
# - Tokenization: split into tokens
# - Stopword removal: remove common English stopwords
# - Lemmatization: reduce words to base form

 
file_id = "austen-emma.txt"  # change to any file from the list
raw_text = gutenberg.raw(file_id)

print("File:", file_id)
print("Characters:", len(raw_text))
print(raw_text[:500])


import re

def clean_and_normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)      # letters/spaces only
    text = re.sub(r"\s+", " ", text).strip()   # collapse extra spaces
    return text

clean_text = clean_and_normalize(raw_text)

print("Raw preview:", raw_text[:150])
print("Clean preview:", clean_text[:150])

 
from nltk.tokenize import word_tokenize

tokens = word_tokenize(clean_text)
print("Token count:", len(tokens))
print(tokens[:30])


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english")).union({"mr", "mrs", "miss"})
tokens_no_stop = [t for t in tokens if t not in stop_words]

lemmatizer = WordNetLemmatizer()
tokens_final = [lemmatizer.lemmatize(t) for t in tokens_no_stop if len(t) > 2]

print("Final token count:", len(tokens_final))
print(tokens_final[:30])


total_tokens = len(tokens_final)
unique_tokens = len(set(tokens_final))
lex_div = unique_tokens / total_tokens

print("Total tokens:", total_tokens)
print("Unique tokens:", unique_tokens)
print("Lexical diversity (unique/total):", round(lex_div, 4))


# ## 4. EDA Findings - Word frequencies and patterns
# 
# - The most frequent words are dominated by character names (examples: emma, harriet, knightley), showing the text focuses on recurring characters and interactions.
# - Dialogue and narration terms (examples: said, think, know) appear often, which makes sense given the text.
# - After preprocessing there were 70,092 tokens and 6,243 unique tokens. Lexical diversity was 0.0891, indicating repeated use of a core vocabulary.

 
from collections import Counter
import pandas as pd

word_freq = Counter(tokens_final)
common_words = word_freq.most_common(20)

df_common = pd.DataFrame(common_words, columns=["word", "count"])
df_common

print("Top 10 words:", df_common.head(10).to_dict(orient="records"))


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.bar(df_common["word"], df_common["count"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 20 Most Frequent Words (Emma)")
plt.xlabel("Word")
plt.ylabel("Count")
plt.show()


from wordcloud import WordCloud

wc_text = " ".join(tokens_final)
wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(wc_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud (Emma)")
plt.show()

 
# ## 5. Word Embedding
# In this section, I convert words into numerical vectors based on context. Sentence tokenization must be performed on the original raw text (with punctuation) so the text can be split into sentences correctly.

 
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import re

sentences_raw = sent_tokenize(raw_text)  # IMPORTANT: raw_text keeps punctuation for sentence boundaries

def prep_sentence(s):
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = word_tokenize(s)
    toks = [t for t in toks if t not in stop_words]
    toks = [lemmatizer.lemmatize(t) for t in toks]
    toks = [t for t in toks if len(t) > 2]
    return toks

sentences = [prep_sentence(s) for s in sentences_raw]
sentences = [s for s in sentences if len(s) > 3]

print("Number of sentences:", len(sentences))
print("Example sentence tokens:", sentences[0][:20])


 
print("Word2Vec settings: vector_size=100, window=5, min_count=2")

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

print("Trained. Vocab size:", len(w2v_model.wv.index_to_key))


test_word = "emma"

print("Most similar to", test_word, ":")
for word, score in w2v_model.wv.most_similar(test_word, topn=10):
    print(f"  {word}: {score:.4f}")

print("\nVector length:", len(w2v_model.wv[test_word]))
print("First 10 values:", w2v_model.wv[test_word][:10])


# ## Word2Vec Results
# 
# - The model returns words that tend to appear in similar contexts to emma in the text. For example, woodhouse and weston appear highly similar, which makes sense because these terms frequently occur near emma throughout the novel.
# - Other words like harriet, feeling, really, and nothing also appear because they often co-occur within nearby context windows.
# - The embedding represents each word as a 100-dimensional numeric vector. The sample values printed above are the first 10 dimensions of the vector for emma.


