import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import bm25s
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CONFIG = {
    "global_top_k": 5,
    "global_bm25_weight": 0.8,
    "global_tfidf_weight": 0.2,
    "multi_answer_margin": 0.85,
    "use_history_bonus": True,
    "history_phrase_bonus": 0.04,
    "history_year_bonus": 0.03,
}

BASE_DIR = Path(__file__).resolve().parent
CHOICES = ["A", "B", "C", "D"]
CANDIDATE_K = 50
LOCAL_EPSILON = 0.25
ANSWER_BIAS_CHOICE = None
ANSWER_BIAS_MARGIN = 0.95


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


def normalize_for_match(text):
    return " ".join(tokenize(text))


def strip_accents(text):
    normalized = unicodedata.normalize("NFD", str(text).lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def extract_time_markers(text):
    text = strip_accents(text)
    markers = set(re.findall(r"\b\d{3,4}\b", text))
    markers.update(re.findall(r"\bthe\s+k[yi]\s+(?:[ivxlcdm]+|\d{1,2})\b", text))
    return markers


def normalize_dataset_items(data):
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        for key in ("data", "documents", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]

    return []


def load_dataset_items(path):
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        first_char = ""
        while True:
            char = f.read(1)
            if char == "":
                return []
            if not char.isspace():
                first_char = char
                break

        f.seek(0)
        if first_char == "[":
            return normalize_dataset_items(json.load(f))

        if first_char == "{":
            f.readline()
            next_char = ""
            while True:
                char = f.read(1)
                if char == "" or not char.isspace():
                    next_char = char
                    break

            f.seek(0)
            if next_char not in ("", "{"):
                try:
                    return normalize_dataset_items(json.load(f))
                except json.JSONDecodeError:
                    f.seek(0)

        items = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.extend(normalize_dataset_items(json.loads(line)))
            except json.JSONDecodeError:
                continue

    return items


def load_corpus(path="dataset.json"):
    docs = []
    for item in load_dataset_items(path):
        title = item.get("title", "")
        content = item.get("content", "")
        docs.append(f"{title} {content}".strip())
    return docs


def minmax(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    if max_value == min_value:
        return np.zeros_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


def build_global_index(docs):
    tokenized_docs = [tokenize(doc) for doc in docs]
    bm25_index = bm25s.BM25(k1=1.5, b=0.75)
    bm25_index.index(tokenized_docs, show_progress=False)

    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_vectors = tfidf_vectorizer.fit_transform(docs)

    return {
        "bm25": bm25_index,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_vectors": tfidf_vectors,
    }


def retrieve_docs(index, docs, query):
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    k = min(CANDIDATE_K, len(docs))
    if k == 0:
        return []

    candidate_ids, bm25_scores = index["bm25"].retrieve(
        [query_tokens],
        k=k,
        show_progress=False,
    )
    candidate_ids = np.array(candidate_ids[0], dtype=int)
    bm25_scores = np.array(bm25_scores[0], dtype=float)

    if candidate_ids.size == 0:
        return []

    query_vector = index["tfidf_vectorizer"].transform([query])
    tfidf_scores = cosine_similarity(
        query_vector,
        index["tfidf_vectors"][candidate_ids],
    ).flatten()

    hybrid_scores = (
        CONFIG["global_bm25_weight"] * minmax(bm25_scores)
        + CONFIG["global_tfidf_weight"] * minmax(tfidf_scores)
    )

    top_n = min(CONFIG["global_top_k"], candidate_ids.size)
    top_local = np.argsort(hybrid_scores)[-top_n:][::-1]

    return [
        docs[int(candidate_ids[i])]
        for i in top_local
        if bm25_scores[i] > 0
    ]


def build_local_bm25(tokenized_docs):
    doc_lens = [len(doc) for doc in tokenized_docs]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0
    postings = defaultdict(list)
    doc_freq = defaultdict(int)

    for doc_id, tokens in enumerate(tokenized_docs):
        counts = Counter(tokens)
        for term, tf in counts.items():
            postings[term].append((doc_id, tf))
            doc_freq[term] += 1

    idf = {}
    idf_sum = 0.0
    negative_terms = []
    n_docs = len(tokenized_docs)

    for term, df in doc_freq.items():
        value = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        idf[term] = value
        idf_sum += value
        if value < 0:
            negative_terms.append(term)

    avg_idf = idf_sum / len(idf) if idf else 0.0
    for term in negative_terms:
        idf[term] = LOCAL_EPSILON * avg_idf

    return {
        "avgdl": avgdl,
        "doc_lens": doc_lens,
        "postings": postings,
        "idf": idf,
    }


def local_bm25_score(local_index, query_tokens):
    if not query_tokens or not local_index["avgdl"]:
        return 0.0

    score = 0.0
    k1 = 1.5
    b = 0.75
    avgdl = local_index["avgdl"]

    for term in query_tokens:
        idf = local_index["idf"].get(term, 0.0)
        if idf == 0.0:
            continue

        for doc_id, tf in local_index["postings"].get(term, []):
            doc_len = local_index["doc_lens"][doc_id]
            denom = tf + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (tf * (k1 + 1) / denom)

    return score


def apply_history_bonus(scores, question_text, question, context, sentences):
    if not CONFIG["use_history_bonus"]:
        return scores

    best_score = max(scores.values())
    if best_score <= 0:
        return scores

    context_norm = f" {normalize_for_match(context)} "
    context_times = extract_time_markers(context)
    question_times = extract_time_markers(question_text)
    sentence_infos = [
        (f" {normalize_for_match(sentence)} ", extract_time_markers(sentence))
        for sentence in sentences
    ]

    adjusted_scores = dict(scores)
    for choice in CHOICES:
        choice_text = question.get(choice, "")
        choice_norm = normalize_for_match(choice_text)
        choice_tokens = choice_norm.split()
        choice_times = extract_time_markers(choice_text)

        bonus_ratio = 0.0
        phrase_found = len(choice_tokens) >= 2 and f" {choice_norm} " in context_norm
        if phrase_found:
            bonus_ratio += CONFIG["history_phrase_bonus"]

        if choice_times and choice_times & context_times:
            bonus_ratio += CONFIG["history_year_bonus"]

        if phrase_found and question_times:
            for sentence_norm, sentence_times in sentence_infos:
                if f" {choice_norm} " in sentence_norm and question_times & sentence_times:
                    bonus_ratio += CONFIG["history_year_bonus"]
                    break

        if bonus_ratio:
            adjusted_scores[choice] += best_score * bonus_ratio

    return adjusted_scores


def select_answers(scores):
    bias_choice = ANSWER_BIAS_CHOICE
    best_choice = max(scores, key=scores.get)
    best_score = scores[best_choice]

    if best_score <= 0:
        return [best_choice]

    threshold = best_score * CONFIG["multi_answer_margin"]
    answers = [
        choice
        for choice in CHOICES
        if scores[choice] > 0 and scores[choice] >= threshold
    ]

    if bias_choice in CHOICES:
        bias_score = scores.get(bias_choice, float("-inf"))
        if bias_score >= best_score * ANSWER_BIAS_MARGIN and bias_choice not in answers:
            answers.append(bias_choice)

    return answers or [best_choice]


def choose_answers(question, docs):
    context = " ".join(docs)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.;!?\n]+", context)
        if sentence.strip()
    ]
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]
    tokenized_sentences = [tokens for tokens in tokenized_sentences if tokens]

    if not tokenized_sentences:
        return ["A"]

    local_index = build_local_bm25(tokenized_sentences)
    question_text = question.get("question", "")
    question_score = local_bm25_score(local_index, tokenize(question_text))

    scores = {}

    for choice in CHOICES:
        query = f"{question_text} {question.get(choice, '')}"
        scores[choice] = local_bm25_score(local_index, tokenize(query)) - question_score

    scores = apply_history_bonus(scores, question_text, question, context, sentences)
    return select_answers(scores)


def make_submission(
    test_file="de_thi.json",
    corpus_file="dataset.json",
    output_file="submission.json",
    zip_file="submission.zip",
):
    test_file = resolve_path(test_file)
    output_file = resolve_path(output_file)
    zip_file = resolve_path(zip_file)

    docs = load_corpus(corpus_file)
    global_index = build_global_index(docs)

    with open(test_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    predictions = []
    for question in questions:
        question_text = question.get("question", "")
        all_options = " ".join(question.get(choice, "") for choice in CHOICES)
        retrieved_docs = retrieve_docs(global_index, docs, f"{question_text} {all_options}")
        answer = choose_answers(question, retrieved_docs)
        predictions.append({"id": question.get("id"), "answer": answer})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname=output_file.name)
        zipf.write(__file__, arcname=Path(__file__).name)

    print(f"Submission created: {zip_file}")


if __name__ == "__main__":
    make_submission()
