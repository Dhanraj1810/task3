import pandas as pd
import numpy as np
import re
from nltk.util import ngrams
from nltk.corpus import words
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from wordcloud import WordCloud
import nltk
nltk.download('words')
data = [
    "The quick brown fox jumps over the lazy dog",
    "Artificial Intelligence and Machine Learning are revolutionizing industries",
    "Python programming is fun and powerful",
    "Natural Language Processing involves working with human language data"
]
text_data = " ".join(data)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\d+', '', text)      
    return text
cleaned_text = preprocess_text(text_data)
tokens = cleaned_text.split()
def generate_ngrams(tokens, n=3):
    n_grams = list(ngrams(tokens, n))
    n_gram_freq = Counter(n_grams)
    return n_gram_freq
n_grams = generate_ngrams(tokens, n=3)
def autocomplete(input_text, n_grams, top_n=3):
    input_tokens = tuple(input_text.lower().split())
    suggestions = [
        " ".join(ngram[len(input_tokens):]) 
        for ngram in n_grams.keys() 
        if ngram[:len(input_tokens)] == input_tokens
    ]
    return suggestions[:top_n]
input_text = "natural language"
autocomplete_suggestions = autocomplete(input_text, n_grams)
print(f"Suggestions for '{input_text}': {autocomplete_suggestions}")
def autocorrect(word, dictionary, top_n=3):
    scores = [(candidate, fuzz.ratio(word, candidate)) for candidate in dictionary]
    scores = sorted(scores, key=lambda x: -x[1])[:top_n]
    return [w for w, _ in scores]
dictionary = set(words.words())
misspelled_word = "languge"
corrections = autocorrect(misspelled_word, dictionary)
print(f"Corrections for '{misspelled_word}': {corrections}")
true_texts = ["language", "processing"]
predicted_texts = ["language", "prosessing"]
accuracy = accuracy_score(true_texts, predicted_texts)
precision = precision_score(true_texts, predicted_texts, average='weighted', zero_division=0)
recall = recall_score(true_texts, predicted_texts, average='weighted', zero_division=0)
f1 = f1_score(true_texts, predicted_texts, average='weighted', zero_division=0)
print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
wordcloud = WordCloud(background_color="white", max_words=50).generate(cleaned_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Text Data")
plt.show()
top_n_grams = n_grams.most_common(10)
ngrams_df = pd.DataFrame(top_n_grams, columns=["n-gram", "Frequency"])
ngrams_df.reset_index(drop=True, inplace=True)
ngrams_df['n-gram'] = ngrams_df['n-gram'].apply(lambda x: ' '.join(x))
plt.figure(figsize=(12, 6))
sns.barplot(data=ngrams_df, x="Frequency", y="n-gram", hue="n-gram", palette="viridis", legend=False)
plt.title("Top 10 n-grams by Frequency")
plt.xlabel("Frequency")
plt.ylabel("n-gram")
plt.tight_layout()
plt.show()

