import json
import re
import zipfile
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CONFIG = {
    "global_top_k": 5,
    "local_top_k": 3,
    "multi_answer_margin": 0.85,
}

BASE_DIR = Path(__file__).resolve().parent
CHOICES = ["A", "B", "C", "D"]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def make_vectorizer():
    return TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


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


def build_tfidf_index(texts):
    if not texts:
        return None

    vectorizer = make_vectorizer()
    try:
        vectors = vectorizer.fit_transform(texts)
    except ValueError:
        return None

    return {
        "vectorizer": vectorizer,
        "vectors": vectors,
    }


def tfidf_scores(index, query):
    if index is None:
        return np.array([], dtype=float)
    query_vector = index["vectorizer"].transform([query])
    return cosine_similarity(query_vector, index["vectors"]).flatten()


def retrieve_docs(index, docs, query):
    scores = tfidf_scores(index, query)
    if scores.size == 0:
        return []

    top_n = min(CONFIG["global_top_k"], len(docs))
    top_ids = np.argsort(scores)[-top_n:][::-1]

    return [
        docs[int(doc_id)]
        for doc_id in top_ids
        if scores[int(doc_id)] > 0
    ]


def split_sentences(docs):
    context = " ".join(docs)
    return [
        sentence.strip()
        for sentence in re.split(r"[.;!?\n]+", context)
        if sentence.strip()
    ]


def local_tfidf_score(index, query):
    scores = tfidf_scores(index, query)
    if scores.size == 0:
        return 0.0

    top_n = min(CONFIG["local_top_k"], scores.size)
    top_scores = np.sort(scores)[-top_n:]
    return float(top_scores.sum())


def select_answers(scores):
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

    return answers or [best_choice]


def choose_answers(question, docs):
    sentences = split_sentences(docs)
    local_index = build_tfidf_index(sentences)
    if local_index is None:
        return ["A"]

    question_text = question.get("question", "")
    question_score = local_tfidf_score(local_index, question_text)

    scores = {}
    for choice in CHOICES:
        query = f"{question_text} {question.get(choice, '')}"
        scores[choice] = local_tfidf_score(local_index, query) - question_score

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
    global_index = build_tfidf_index(docs)

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
