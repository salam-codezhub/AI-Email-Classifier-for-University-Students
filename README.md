# 📧 AI Email Classifier for University Students

> An NLP-powered email categorisation system built with Python, scikit-learn, and NLTK — designed as a beginner-friendly college project.

---

## 🗂️ Project Overview

University students receive hundreds of emails every semester — fest invites, club announcements, placement drives, research calls, and general notices all land in the same inbox. This project builds an **AI/ML text classifier** that automatically sorts incoming emails into five categories:

| Category | Examples |
|---|---|
| 🎉 **Fest** | Hackathons, cultural fests, sports week, TechFest |
| 🏛️ **Club** | Club recruitment, weekly sessions, drama auditions |
| 💼 **Placement** | Campus drives, internship openings, resume workshops |
| 🔬 **Research** | Paper calls, PhD admissions, workshop on algorithms |
| 📋 **Others** | Exam schedules, fee reminders, holiday notices |

---

## 🧠 How It Works (AI/ML Concepts)

```
Raw Email (Subject + Body)
        │
        ▼
  ┌─────────────────────┐
  │  Text Preprocessing │  ← lowercase, remove punctuation,
  │                     │    stopword removal, stemming
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │   TF-IDF Vectoriser │  ← converts text into numeric features
  │   (500 features,    │    weights rare-but-important words higher
  │    unigrams+bigrams)│
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │ Logistic Regression │  ← multi-class classifier, learns
  │    Classifier       │    word-to-category associations
  └─────────┬───────────┘
            │
            ▼
  Predicted Category + Confidence Scores
```

### Key Concepts Demonstrated

- **NLP Pipeline** — tokenisation → stopword removal → stemming
- **TF-IDF** — Term Frequency × Inverse Document Frequency feature extraction
- **Logistic Regression** — supervised multi-class text classification
- **Train/Test Split** — 80/20 stratified split for unbiased evaluation
- **Classification Report** — precision, recall, F1-score per category

---

## 📁 Repository Structure

```
ai-email-classifier/
│
├── email_classifier.py     ← Main source file (single runnable script)
├── requirements.txt        ← Python dependencies
├── .gitignore              ← Files to exclude from Git
└── README.md               ← This file
```

---

## ⚙️ Requirements

- Python 3.8 or higher
- pip (Python package manager)

### Dependencies

```
scikit-learn
pandas
nltk
```

---

## 🚀 Setup & Run

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/ai-email-classifier.git
cd ai-email-classifier
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the classifier

```bash
python email_classifier.py
```

### Expected Output

```
============================================================
   AI Email Classifier for University Students
============================================================

[1/3] Loading and preprocessing dataset...
      Total emails in dataset : 30
      Categories              : ['Club', 'Fest', 'Others', 'Placement', 'Research']

[2/3] Training TF-IDF + Logistic Regression model...

      ✅ Model Accuracy : 100.0%

      Detailed Classification Report:
              precision    recall  f1-score   support
        Club       1.00      1.00      1.00         1
        Fest       1.00      1.00      1.00         1
      Others       1.00      1.00      1.00         2
   Placement       1.00      1.00      1.00         1
    Research       1.00      1.00      1.00         1

[3/3] Entering interactive mode. Type 'quit' to exit.

------------------------------------------------------------
Enter email SUBJECT (or 'quit'): Amazon hiring drive on campus
Enter email BODY    : Amazon will visit campus next week for SDE intern positions. Eligible branches: CSE, IT. Register now.

  📧 Predicted Category  :  ★ Placement ★
  Confidence scores:
    Placement            ████████████████     82.3%
    Others               ██                   9.1%
    Research             █                    4.8%
    Club                 █                    2.1%
    Fest                                       1.7%
```

---

## 🔬 Sample Emails to Try

**Fest:**
> Subject: `Annual Hackathon – Register Now`
> Body: `Join us for a 24-hour coding competition with prizes worth ₹50,000. Form teams and register on the portal.`

**Club:**
> Subject: `Photography Club New Member Drive`
> Body: `We are recruiting new members for the photography club. No prior experience needed. Come to Room 204 on Saturday.`

**Placement:**
> Subject: `Google STEP Internship – Applications Open`
> Body: `Google is offering summer internships for first and second year students. Strong coding skills required. Apply now.`

**Research:**
> Subject: `Call for Papers – IEEE Conference`
> Body: `We invite original research papers on AI and Machine Learning. Submission deadline is 1st May. Double-blind peer review.`

**Others:**
> Subject: `End Semester Exam Schedule Released`
> Body: `The examination timetable has been published. Please check the academic portal for your individual schedule.`

---

## 📊 Model Details

| Component | Choice | Why |
|---|---|---|
| Feature Extraction | TF-IDF (500 features, 1–2 grams) | Weighs domain-specific words higher than common filler words |
| Classifier | Logistic Regression | Fast, interpretable, strong baseline for text classification |
| Preprocessing | Lowercase + punctuation removal + stopword filter + Porter Stemmer | Reduces vocabulary noise and groups word variants |
| Split | 80% train / 20% test (stratified) | Ensures all categories appear in the test set |

---

## 📜 License

This project is released under the MIT License — free to use, modify, and distribute for academic purposes.

---

## 👤 Author
SALAM KHAN

