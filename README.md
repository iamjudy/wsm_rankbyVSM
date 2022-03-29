# Ranking by Vector Space Model

It's a retrieval program that is able to retrieve the relevant news to the given query from a set of 7,034 English News collected from reuters.com according to different weighting schemes and similarity metrics.

## Guide

Basic command looks like:

`$ python3 main.py -d <doc> -w <weightType> -r <relevanceType> -q <query>`

Arguments:
- `-d`, `--doc`: The file path of documents which are ready to be ranked.
- `-w`, `--weightType`: **tf** or **tfidf**.
- `-r`, `--relevanceType`: **cos** for 'cosine similarity' or **eu** for 'Euclidean Distance'.
- `-q`, `--query`: input the searching query, default value is **'Trump Biden Taiwan China'**.
  

There are four combinations can be chose:

#### 1. Term Frequency (TF) Weighting + Cosine Similarity.
  ```bash
  python3 main.py -w tf -r cos
  ```
 #### 2. Term Frequency (TF) Weighting + Euclidean Distance.
  ```bash
  python3 main.py -w tf -r eu
  ```
  #### 3. TF-IDF Weighting + Cosine Similarity.
  ```bash
  python3 main.py -w tfidf -r cos
  ```
  #### 4. TF-IDF Weighting + Euclidean Distance.
  ```bash
  python3 main.py -w tfidf -r eu
  ```

  ---
### Relevance Feedback
Relevance Feedback is an IR technique for improving retrieved results. The simplest approach is Pseudo Feedback, the idea of which is to feed the results retrieved by the given query, and then to use the content of the fed results as supplement queries to re-score the documents.

```bash
  python3 main.py -w tfidf -r cos -f
  ```

---
### Bonus : VSM with Different Scheme & Similarity Metrics in Chinese and English
```bash
  python3 main.py -d News -w tfidf -r cos -q 烏克蘭大選
```