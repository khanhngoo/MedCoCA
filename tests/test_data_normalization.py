from coca_med.data.features import add_engineered_features
from coca_med.data.medqa import normalize_medqa_row
from coca_med.data.pubmedqa import normalize_pubmedqa_row


def test_normalize_medqa_row() -> None:
    row = {
        "question": "Which antibiotic is appropriate?",
        "answer": "Nitrofurantoin",
        "options": {
            "A": "Ampicillin",
            "B": "Nitrofurantoin",
            "C": "Ciprofloxacin",
            "D": "Doxycycline",
        },
        "meta_info": "step2",
        "answer_idx": "B",
        "metamap_phrases": ["antibiotic", "pregnancy"],
    }

    example = normalize_medqa_row(row, split="train", index=7)

    assert example.id == "medqa:train:7"
    assert example.dataset == "medqa"
    assert example.gold_label == "B"
    assert example.gold_answer == "Nitrofurantoin"
    assert example.choices["D"] == "Doxycycline"


def test_normalize_pubmedqa_row() -> None:
    row = {
        "pubid": 123,
        "question": "Does treatment improve survival?",
        "context": {
            "contexts": ["A trial was conducted.", "Survival improved."],
            "labels": ["BACKGROUND", "RESULTS"],
            "meshes": ["Survival"],
        },
        "long_answer": "Treatment improved survival in this cohort.",
        "final_decision": "yes",
    }

    example = normalize_pubmedqa_row(row, config="pqa_labeled", split="train", index=3)

    assert example.id == "pubmedqa/pqa_labeled:train:3"
    assert example.dataset == "pubmedqa/pqa_labeled"
    assert example.gold_label == "yes"
    assert "BACKGROUND" in example.context
    assert example.choices == {"yes": "yes", "no": "no", "maybe": "maybe"}


def test_add_engineered_features() -> None:
    example = normalize_medqa_row(
        {
            "question": "A short medical question?",
            "answer": "A",
            "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
            "answer_idx": "A",
            "metamap_phrases": ["medical"],
        }
    )

    featured = add_engineered_features(example)

    assert featured.features["question_words"] > 0
    assert featured.features["num_choices"] == 4
    assert featured.features["difficulty_proxy"] > 0
