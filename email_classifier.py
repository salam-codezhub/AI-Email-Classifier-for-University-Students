# =============================================================================
# AI Email Classifier for University Students
# =============================================================================
# Author  : College Project Template
# Language: Python 3.x
# Libraries: scikit-learn, pandas, nltk
#
# HOW TO RUN:
#   1. Install dependencies:
#        pip install -r requirements.txt
#   2. Run the script:
#        python email_classifier.py
#   3. The program will train the model, print accuracy, then ask you to
#      enter an email subject + body for live prediction.
# =============================================================================

# ── Standard / Third-party imports ──────────────────────────────────────────
import re
import string

import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Download required NLTK data (runs once, cached locally) ─────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)


# =============================================================================
# SECTION 1: DATASET
# Each entry is a tuple of (subject, body, label).
# Labels: Fest | Club | Placement | Research | Others
# 30 hand-crafted examples representing real university email types.
# =============================================================================

RAW_DATA = [
    # ── Fest (6 examples) ─────────────────────────────────────────────────
    (
        "Techfest 2024 – Register Now!",
        "Dear student, Techfest is back with exciting competitions, workshops and cultural nights. "
        "Register before 30th March.",
        "Fest"
    ),
    (
        "Annual Cultural Fest – Utsav",
        "Join us for three days of music, dance, drama and food stalls. "
        "All students are welcome to participate.",
        "Fest"
    ),
    (
        "Hackathon at TechFest",
        "24-hour coding hackathon. Form teams of 3. Prizes worth Rs 50,000. "
        "Last date to register is this Friday.",
        "Fest"
    ),
    (
        "Farewell Fest for Final Year Students",
        "The student council invites all final year students to the grand farewell night. "
        "RSVP by Tuesday.",
        "Fest"
    ),
    (
        "Robowar and Coding Contest – TechNova",
        "Gear up for TechNova! Events include Robowar, Quiz, Code Sprint and Photography. "
        "Register now on the portal.",
        "Fest"
    ),
    (
        "Sports Week Inauguration",
        "The annual sports week begins Monday. Events: cricket, football, badminton, chess. "
        "Represent your department!",
        "Fest"
    ),

    # ── Club (6 examples) ─────────────────────────────────────────────────
    (
        "Photography Club – New Member Recruitment",
        "The Photography Club is recruiting new members for the academic year. "
        "No prior experience required. Meet us at Room 204.",
        "Club"
    ),
    (
        "Coding Club Weekly Session",
        "This week's coding club session covers dynamic programming. "
        "Venue: CS Lab 3. Time: 5 PM Saturday.",
        "Club"
    ),
    (
        "Drama Club Auditions",
        "Auditions for the annual play are open. We need actors, directors and backstage crew. "
        "Walk in on Thursday.",
        "Club"
    ),
    (
        "Entrepreneurship Cell – Monthly Meetup",
        "E-Cell monthly meetup this Sunday. Guest speaker from a Series-B startup will share "
        "their founding story.",
        "Club"
    ),
    (
        "Literary Club – Debate Competition",
        "The Literary Club is organising an inter-department debate. "
        "Topic: AI will replace humans. Sign up by Friday.",
        "Club"
    ),
    (
        "Music Club Practice Session",
        "Weekly practice session for band members on Saturday at 4 PM in the amphitheatre. "
        "New vocalists welcome.",
        "Club"
    ),

    # ── Placement / Internship (6 examples) ───────────────────────────────
    (
        "Amazon SDE Internship – Campus Drive",
        "Amazon will be visiting campus on 15th April for SDE intern roles. "
        "Eligible: 3rd year CSE/IT. Register on the placement portal.",
        "Placement"
    ),
    (
        "Infosys Off-Campus Placement Drive",
        "Infosys is hiring fresher engineers. Apply through the career portal. "
        "Eligible branches: CSE, ECE, EEE, IT.",
        "Placement"
    ),
    (
        "Summer Internship – Microsoft Research",
        "Microsoft Research is offering paid 10-week summer internships. "
        "Apply with your CV and a statement of purpose.",
        "Placement"
    ),
    (
        "Goldman Sachs – Finance Internship Opportunity",
        "Goldman Sachs is offering summer analyst internships for finance and quantitative roles. "
        "Deadline: 20th March.",
        "Placement"
    ),
    (
        "Resume Building Workshop by Placement Cell",
        "The placement cell is conducting a resume building and LinkedIn optimisation workshop. "
        "Attendance is recommended.",
        "Placement"
    ),
    (
        "Google STEP Internship – Applications Open",
        "Google's STEP Internship program for first and second year students is now open. "
        "Strong coding skills required.",
        "Placement"
    ),

    # ── Research / Conference (6 examples) ───────────────────────────────
    (
        "Call for Papers – IEEE Conference on AI",
        "We invite original research papers on AI, ML and Data Science. "
        "Submission deadline: 1st May. Double-blind review.",
        "Research"
    ),
    (
        "Research Internship at IIT Delhi",
        "IIT Delhi's CSE department offers summer research internships in NLP and computer vision. "
        "Apply with CV and SOP.",
        "Research"
    ),
    (
        "National Conference on Renewable Energy",
        "Abstract submissions open for NCRE 2024. Topics: solar, wind, hydrogen fuel cells. "
        "Proceedings will be indexed.",
        "Research"
    ),
    (
        "PhD Admissions – Applications Open",
        "Applications are invited for the PhD programme in Computer Science and Engineering. "
        "Specialisations: AI, Systems, Theory.",
        "Research"
    ),
    (
        "Paper Acceptance – ICML 2024",
        "Congratulations! Your paper has been accepted at ICML 2024. "
        "Please submit the camera-ready version by 10th June.",
        "Research"
    ),
    (
        "Workshop on Quantum Computing – Registrations Open",
        "A two-day workshop on quantum computing algorithms will be held at the department. "
        "Faculty and students can register.",
        "Research"
    ),

    # ── Others (6 examples) ───────────────────────────────────────────────
    (
        "Exam Schedule for End Semester",
        "The end semester examination schedule has been released. "
        "Please check the academic portal for your timetable.",
        "Others"
    ),
    (
        "Library Fee Payment Reminder",
        "This is a reminder to pay your pending library dues before the end of the month "
        "to avoid a fine.",
        "Others"
    ),
    (
        "Holiday Notice – Diwali Break",
        "The institute will remain closed from 10th to 14th November on account of Diwali. "
        "Classes resume on 15th.",
        "Others"
    ),
    (
        "Hostel Maintenance – Water Supply Disruption",
        "Water supply in Block C and D will be disrupted on Sunday from 6 AM to 2 PM "
        "due to maintenance work.",
        "Others"
    ),
    (
        "Scholarship Application Deadline",
        "Students who wish to apply for the merit-cum-means scholarship must submit their "
        "forms by 25th March.",
        "Others"
    ),
    (
        "COVID Vaccination Drive on Campus",
        "A free vaccination drive is being organised on campus. "
        "All eligible students and staff should participate.",
        "Others"
    ),
]


# =============================================================================
# SECTION 2: TEXT PREPROCESSING
#
# Pipeline:
#   1. Lowercase all text
#   2. Remove digits and punctuation
#   3. Split into tokens (words)
#   4. Remove common English stopwords (e.g. "the", "is", "and")
#   5. Apply Porter Stemming to reduce words to their root
#      (e.g. "registering" → "registr", "competitions" → "competit")
#
# Why preprocess?
#   Raw text is noisy. Preprocessing reduces vocabulary size, groups
#   word variants together, and removes words that carry no category signal.
# =============================================================================

stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Clean and normalise a raw text string.

    Parameters
    ----------
    text : str  – raw email subject + body

    Returns
    -------
    str  – space-joined list of cleaned, stemmed tokens
    """
    # Step 1: Convert everything to lowercase
    text = text.lower()

    # Step 2: Remove all digits (numbers rarely help classify)
    text = re.sub(r"\d+", "", text)

    # Step 3: Remove punctuation characters
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 4: Split into individual word tokens
    tokens = text.split()

    # Step 5: Remove stopwords and very short tokens (length <= 2)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    # Step 6: Stem each token to its root form
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)


# =============================================================================
# SECTION 3: BUILD DATAFRAME AND APPLY PREPROCESSING
# =============================================================================

def build_dataset(raw: list) -> pd.DataFrame:
    """Convert raw list of tuples into a preprocessed DataFrame.

    Parameters
    ----------
    raw : list of (subject, body, label) tuples

    Returns
    -------
    pd.DataFrame with columns: subject, body, label, text, clean_text
    """
    df = pd.DataFrame(raw, columns=["subject", "body", "label"])

    # Combine subject and body — more text gives the model a stronger signal
    df["text"] = df["subject"] + " " + df["body"]

    # Apply the preprocessing pipeline to every email
    df["clean_text"] = df["text"].apply(preprocess)

    return df


# =============================================================================
# SECTION 4: FEATURE EXTRACTION AND MODEL TRAINING
#
# We use a classic NLP pipeline:
#   TF-IDF Vectoriser → Logistic Regression Classifier
#
# TF-IDF (Term Frequency × Inverse Document Frequency):
#   - TF  : how often a word appears in THIS email
#   - IDF : penalises words that appear in ALL emails (common words)
#   - Result: words like "hackathon" score high in Fest emails;
#             words like "the" score near zero everywhere.
#
# Logistic Regression:
#   - A linear classifier that learns a weight for each word-category pair.
#   - Outputs a probability distribution across all categories.
#   - Simple, fast, and highly effective for text.
# =============================================================================

def train_model(df: pd.DataFrame):
    """Train a TF-IDF + Logistic Regression pipeline.

    Parameters
    ----------
    df : pd.DataFrame – preprocessed dataset

    Returns
    -------
    vectorizer : fitted TfidfVectorizer
    model      : fitted LogisticRegression
    accuracy   : float – test set accuracy (0 to 1)
    report     : str   – full per-class classification report
    """
    X = df["clean_text"]   # Features: cleaned email text
    y = df["label"]        # Target: category label

    # ── 80/20 Train-Test Split ────────────────────────────────────────────
    # stratify=y ensures every category appears in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ── TF-IDF Feature Extraction ─────────────────────────────────────────
    # max_features=500  → keep only the 500 most informative word/phrase features
    # ngram_range=(1,2) → use both single words ("internship") and
    #                     two-word phrases ("campus drive") as features
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
    )

    # fit_transform: learn vocabulary from training data AND transform it
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # transform only: apply the SAME vocabulary to test data (no leakage)
    X_test_tfidf  = vectorizer.transform(X_test)

    # ── Logistic Regression Classifier ───────────────────────────────────
    # C=1.0       → regularisation strength (higher = less regularisation)
    # max_iter    → increased from default 100 to ensure convergence
    # random_state → for reproducibility
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred   = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, zero_division=0)

    return vectorizer, model, accuracy, report


# =============================================================================
# SECTION 5: PREDICTION FUNCTION
#
# Given a new email (subject + body), this function:
#   1. Combines and preprocesses the text
#   2. Transforms it using the FITTED vectorizer
#   3. Returns the predicted category + confidence scores
# =============================================================================

def predict_category(subject: str, body: str,
                     vectorizer: TfidfVectorizer,
                     model: LogisticRegression) -> tuple:
    """Predict the email category for a single new email.

    Parameters
    ----------
    subject    : str – email subject line
    body       : str – email body text
    vectorizer : fitted TfidfVectorizer (from training)
    model      : fitted LogisticRegression (from training)

    Returns
    -------
    (predicted_label : str,
     confidence_dict : dict mapping category → percentage)
    """
    # Combine and preprocess exactly as during training
    raw_text   = subject + " " + body
    clean_text = preprocess(raw_text)

    # Vectorise (transform only — do NOT fit again)
    vec = vectorizer.transform([clean_text])

    # Get predicted label and probability scores
    predicted = model.predict(vec)[0]
    proba     = model.predict_proba(vec)[0]
    classes   = model.classes_

    # Build a readable confidence dictionary
    confidence = {
        cls: round(prob * 100, 1)
        for cls, prob in zip(classes, proba)
    }

    return predicted, confidence


# =============================================================================
# SECTION 6: MAIN PROGRAM
#
# Ties everything together:
#   Step 1 → Load and preprocess dataset
#   Step 2 → Train model and print evaluation
#   Step 3 → Interactive prediction loop
# =============================================================================

def main():
    print("=" * 60)
    print("   AI Email Classifier for University Students")
    print("=" * 60)

    # ── Step 1: Dataset ───────────────────────────────────────────────────
    print("\n[1/3] Loading and preprocessing dataset...")
    df = build_dataset(RAW_DATA)
    print(f"      Total emails in dataset : {len(df)}")
    print(f"      Categories              : {sorted(df['label'].unique())}")

    # ── Step 2: Training ──────────────────────────────────────────────────
    print("\n[2/3] Training TF-IDF + Logistic Regression model...")
    vectorizer, model, accuracy, report = train_model(df)

    print(f"\n      ✅ Model Accuracy : {accuracy * 100:.1f}%")
    print("\n      Detailed Classification Report:")
    print(report)

    # ── Step 3: Interactive Prediction ───────────────────────────────────
    print("[3/3] Entering interactive mode. Type 'quit' to exit.\n")

    while True:
        print("-" * 60)
        subject = input("Enter email SUBJECT (or 'quit'): ").strip()

        if subject.lower() == "quit":
            print("Goodbye!")
            break

        body = input("Enter email BODY    : ").strip()

        if not subject and not body:
            print("⚠️  Please enter at least a subject or body text.\n")
            continue

        # Predict and display results
        predicted, confidence = predict_category(
            subject, body, vectorizer, model
        )

        print(f"\n  📧 Predicted Category  :  ★ {predicted} ★")
        print("  Confidence scores:")

        # Display sorted by confidence (highest first) with a bar
        for cat, pct in sorted(confidence.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct / 5)
            print(f"    {cat:<20} {bar:<20} {pct:5.1f}%")

        print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
