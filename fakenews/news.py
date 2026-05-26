from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class NewsItem:
    """Single news example (title + text) with ground-truth label."""

    news_id: int
    title: str
    text: str
    label: str  # "FAKE" or "REAL"


class FakeOrRealNewsCSVReader:
    """Loads the 'Fake or Real News' CSV into structured `NewsItem`s."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def read(self) -> List[NewsItem]:
        items: List[NewsItem] = []
        if not self.csv_path.exists():
            return items

        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Expected columns: title, text, label. The first column is often an index.
                try:
                    news_id = int(row.get("", "") or row.get("id", "") or row.get("index", "") or len(items))
                except ValueError:
                    news_id = len(items)

                title = (row.get("title") or "").strip()
                text = (row.get("text") or "").strip()
                label = (row.get("label") or "").strip().upper()
                if not text or label not in {"FAKE", "REAL"}:
                    continue
                items.append(NewsItem(news_id=news_id, title=title, text=text, label=label))

        return items


class FakeNewsDataset:
    """In-memory dataset + basic helpers."""

    def __init__(self, items: Iterable[NewsItem]) -> None:
        self.items: List[NewsItem] = list(items)

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "FakeNewsDataset":
        reader = FakeOrRealNewsCSVReader(csv_path)
        return cls(reader.read())

    def filter_label(self, label: str) -> List[NewsItem]:
        label = label.upper()
        return [i for i in self.items if i.label.upper() == label]


class FakeNewsClassifier:
    """Binary text classifier (FAKE vs REAL) based on TF-IDF + LogisticRegression."""

    def __init__(self) -> None:
        self._pipeline = None
        self.is_trained = False

    def train(self, dataset: FakeNewsDataset) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        texts = [i.text for i in dataset.items]
        y = [1 if i.label.upper() == "FAKE" else 0 for i in dataset.items]
        if len(set(y)) < 2:
            # Not enough classes.
            self.is_trained = False
            self._pipeline = None
            return

        self._pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(stop_words="english", max_features=30000)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        self._pipeline.fit(texts, y)
        self.is_trained = True

    def predict_proba_fake(self, text: str) -> float:
        if not self.is_trained or self._pipeline is None:
            return 0.5
        proba = self._pipeline.predict_proba([text])[0]
        # class 1 => FAKE
        return float(proba[1])

    def predict_label(self, text: str, threshold: float = 0.5) -> str:
        return "FAKE" if self.predict_proba_fake(text) >= threshold else "REAL"


@dataclass
class NewsPrediction:
    predicted_label: str
    proba_fake: float


def predict_news(classifier: Optional[FakeNewsClassifier], item: Optional[NewsItem]) -> Optional[NewsPrediction]:
    if classifier is None or item is None:
        return None
    p = classifier.predict_proba_fake(item.text)
    return NewsPrediction(predicted_label=("FAKE" if p >= 0.5 else "REAL"), proba_fake=p)
