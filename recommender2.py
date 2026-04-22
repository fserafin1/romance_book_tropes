#import all necessary libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
import re


@st.cache_resource
def load_and_train_model():
    from sentence_transformers import SentenceTransformer
    
    #define and load data
    df = pd.read_csv("/workspaces/romance_book_tropes/romance_books_32K.zip", compression='zip')
<<<<<<< HEAD

    # Sample a smaller subset to reduce memory and time
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
=======
>>>>>>> d9ef346ad5e4ace258c73f9fbda9ccfa8ad43870

    #get trope column as it is binary rather than text
    trope_columns =df.columns[7:]

    #clean data
    df = df.dropna(subset=["description"])
    df = df.reset_index(drop=True)
    df = df.copy()

    #defining full text to be preprocessed
    df["full_text"] = df[["title", "author", "description"]].fillna("").agg(" ".join, axis=1)

    #cleaning up text 
    df["clean_text"] = (
        df["full_text"]
        .str.lower()
        .str.replace(r"[^a-z\s]", "", regex=True)
    )

    #transformers
    #matrix set up
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = df["clean_text"].astype(str).tolist()

    X = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    y = df[trope_columns].to_numpy()

    # Ensure trope_columns is a regular list if it's a pandas Index
    if not isinstance(trope_columns, list):
        trope_columns = trope_columns.tolist()

    #train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = OneVsRestClassifier(
        LGBMClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    )
    clf.fit(X_train, y_train)

    #accuracy
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="micro")
    print("F1 Score:", f1)
    
    return model, clf, trope_columns


#predict tropes
def predict_tropes(title, author, description, model, clf, trope_columns, threshold=0.15, top_k=5):
    text = f"{title} {author} {description}".lower()
    text = re.sub(r"[^a-z\s]", "", text)

    emb = model.encode([text], convert_to_numpy=True)

    probs = clf.predict_proba(emb)[0]

    #normalize to prevent one trope dominating
    if probs.sum() > 0:
        probs = probs / probs.sum()

    #always get top tropes
    top = sorted(
        [(trope_columns[i], p) for i, p in enumerate(probs)],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    #also include anything above threshold
    above_thresh = [
        (trope_columns[i], p)
        for i, p in enumerate(probs)
        if p >= threshold
    ]

    #merge results (avoid duplicates)
    combined = {t: p for t, p in top}
    for t, p in above_thresh:
        combined[t] = p

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


