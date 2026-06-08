import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CONFIG = {
    "threshold_ratio": 0.1,
    "top_k_docs": 20,
    "use_choices_in_query": True,
    "use_rank_weight": True,
}

BASE_DIR = Path(__file__).resolve().parent
CHOICES = ["A", "B", "C", "D"]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def tokenize(text):
    if not text:
        return []
    return re.findall(r"\w+", str(text).lower())


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def load_dataset_items(path):
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
    else:
        data = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return [item for item in data if isinstance(item, dict)]


def load_corpus(corpus_file="dataset.json"):
    documents = []
    doc_token_sets = []

    try:
        data = load_dataset_items(corpus_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return [], []

    for doc in data:
        text = f"{doc.get('title', '')} {doc.get('content', '')}".strip()
        documents.append(text)
        doc_token_sets.append(set(tokenize(text)))

    return documents, doc_token_sets


def build_tfidf_index(documents):
    if not documents:
        return None

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    try:
        vectors = vectorizer.fit_transform(documents)
    except ValueError:
        return None

    return {
        "vectorizer": vectorizer,
        "vectors": vectors,
    }


def search(index, doc_token_sets, query, k):
    if index is None:
        return []

    query_vector = index["vectorizer"].transform([query])
    scores = cosine_similarity(query_vector, index["vectors"]).flatten()
    if scores.size == 0:
        return []

    top_n = min(k, scores.size)
    top_ids = np.argsort(scores)[-top_n:][::-1]

    return [
        {
            "tokens": doc_token_sets[int(doc_id)],
            "rank": rank,
        }
        for rank, doc_id in enumerate(top_ids)
        if scores[int(doc_id)] > 0
    ]


def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def select_answers(item, candidate_docs, threshold_ratio):
    scores = {}
    max_score = 0.0

    for choice in CHOICES:
        choice_text = item.get(choice, "")
        choice_tokens = tokenize(choice_text)
        best_doc_score = 0.0

        for doc_tokens in candidate_docs:
            rank_weight = 1.0
            if isinstance(doc_tokens, dict):
                rank = doc_tokens.get("rank", 0)
                rank_weight = 1.0 / (rank + 1) if CONFIG["use_rank_weight"] else 1.0
                doc_tokens = doc_tokens["tokens"]

            score = jaccard(set(choice_tokens), doc_tokens) * rank_weight
            if score > best_doc_score:
                best_doc_score = score

        if best_doc_score == 0 and choice_text and candidate_docs:
            best_doc = candidate_docs[0]
            if isinstance(best_doc, dict):
                best_doc = best_doc["tokens"]
            best_doc_text = " ".join(best_doc)
            dist = edit_distance(choice_text.lower(), best_doc_text.lower())
            if best_doc_text:
                norm_dist = dist / max(len(choice_text), len(best_doc_text))
                best_doc_score = 1.0 - norm_dist

        scores[choice] = best_doc_score
        max_score = max(max_score, best_doc_score)

    threshold = threshold_ratio * max_score if max_score > 0 else 0.0
    selected = [
        choice
        for choice in CHOICES
        if scores[choice] >= threshold and scores[choice] > 0
    ]

    return sorted(selected) or [max(scores, key=scores.get)]


def build_query_text(item):
    question_text = item.get("question", "")
    if not CONFIG["use_choices_in_query"]:
        return question_text

    all_choices = " ".join(item.get(choice, "") for choice in CHOICES)
    return f"{question_text} {all_choices}".strip()


def make_submission(
    test_file="de_thi.json",
    corpus_file="dataset.json",
    output_file="submission.json",
    zip_file="submission.zip",
):
    test_file = resolve_path(test_file)
    output_file = resolve_path(output_file)
    zip_file = resolve_path(zip_file)

    documents, doc_token_sets = load_corpus(corpus_file)
    if not documents:
        print("No documents loaded.")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    index = build_tfidf_index(documents)
    submissions = []

    for item in test_data:
        candidate_docs = search(
            index,
            doc_token_sets,
            build_query_text(item),
            CONFIG["top_k_docs"],
        )
        answers = select_answers(item, candidate_docs, CONFIG["threshold_ratio"])
        submissions.append({"id": item.get("id"), "answer": answers})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submissions, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname=os.path.basename(output_file))
        zipf.write(__file__, arcname=os.path.basename(__file__))

    print(f"Processed {len(submissions)} questions.")
    print(f"Submission ZIP created: {zip_file.name}")


if __name__ == "__main__":
    make_submission()
