import os
import sys
import pandas as pd
import numpy as np
import joblib


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# --- Six Emotion Labels ---
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}


# --- Heuristic overrides ---
GREETINGS = ("hi", "hello", "hey", "greetings")
PROFANITY = ("fuck", "shit", "bitch", "damn")


def load_and_merge(data_dir="data"):
    """Load and concatenate the four CSVs, filter to labels 0–5, drop duplicates."""
    files = ["training.csv", "validation.csv", "test.csv", "text.csv"]
    dfs = []
    for fname in files:
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            print(f"ERROR: `{fname}` not found in `{data_dir}`.", file=sys.stderr)
            sys.exit(1)
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["label"].isin(EMOTION_LABELS.keys())]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return df


def split_data(df, seed=42):
    """Stratified 70/15/15 train/val/test split."""
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=seed
    )
    return train, val, test


def train_pipeline(train_df):
    """Fit TF‑IDF and train LogisticRegression; returns (vectorizer, model)."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9
    )
    X_train = vectorizer.fit_transform(train_df["text"].astype(str))
    y_train = train_df["label"]
    clf = LogisticRegression(
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=1000,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return vectorizer, clf


def evaluate(clf, vect, df, split_name="Validation"):
    """Print metrics and confusion matrix."""
    X = vect.transform(df["text"].astype(str))
    y_true = df["label"]
    y_pred = clf.predict(X)


    print(f"\n=== {split_name} Results ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    print(classification_report(
        y_true, y_pred, target_names=list(EMOTION_LABELS.values())
    ))
    cm = confusion_matrix(y_true, y_pred, labels=list(EMOTION_LABELS.keys()))
    cm_df = pd.DataFrame(
        cm,
        index=list(EMOTION_LABELS.values()),
        columns=list(EMOTION_LABELS.values())
    )
    print("Confusion Matrix:")
    print(cm_df)


def save_pipeline(vectorizer, clf, out_path="data/pipeline.pkl"):
    """Save the vectorizer+model together."""
    joblib.dump({"vect": vectorizer, "clf": clf}, out_path)
    print(f"\nSaved both TF‑IDF and classifier to `{out_path}`")


def train_and_export(data_dir="data"):
    # 1) Load & merge
    df = load_and_merge(data_dir)
    # 2) Split
    train_df, val_df, test_df = split_data(df)
    # 3) Train
    vectorizer, clf = train_pipeline(train_df)
    # 4) Evaluate
    evaluate(clf, vectorizer, val_df, split_name="Validation")
    evaluate(clf, vectorizer, test_df,  split_name="Test")
    # 5) Save pipeline
    save_pipeline(vectorizer, clf, out_path=os.path.join(data_dir, "pipeline.pkl"))


# --- Inference (no args) ---
_pipeline = None
def predict_emotion(text: str):
    """
    Returns (predicted_emotion: str, probabilities: Dict[str,float]).
    Loads pipeline.pkl from `data/` on first call.
    Applies heuristic overrides for greetings and profanity.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(os.path.join("data", "pipeline.pkl"))
    vect = _pipeline["vect"]
    clf  = _pipeline["clf"]


    cleaned = text.lower().strip()


    # 1) Greeting override → joy
    for greet in GREETINGS:
        if cleaned == greet or cleaned.startswith(greet + " "):
            # still compute probabilities for display
            probs = clf.predict_proba(vect.transform([cleaned]))[0]
            prob_dict = {EMOTION_LABELS[c]: float(p) for c,p in zip(clf.classes_, probs)}
            return "joy", prob_dict


    # 2) Profanity override → anger
    for bad in PROFANITY:
        if bad in cleaned:
            probs = clf.predict_proba(vect.transform([cleaned]))[0]
            prob_dict = {EMOTION_LABELS[c]: float(p) for c,p in zip(clf.classes_, probs)}
            return "anger", prob_dict


    # 3) Fallback to model prediction
    X = vect.transform([cleaned])
    probs = clf.predict_proba(X)[0]
    idx = np.argmax(probs)
    pred_label = clf.classes_[idx]
    pred_emotion = EMOTION_LABELS[pred_label]
    prob_dict = {EMOTION_LABELS[c]: float(p) for c,p in zip(clf.classes_, probs)}
    return pred_emotion, prob_dict


# --- CLI Entrypoint ---
if __name__ == "__main__":
    train_and_export(data_dir="data")
