# 🛡️ Signal – AI Spam Detector

An end-to-end NLP-powered spam detection web application built using **spaCy**, **Flask**, and a responsive frontend. The application classifies incoming messages as **Spam**, **Authentic**, or **Uncertain** using a custom-trained machine learning model.

---

## 🌐 Live Demo

**Application:**

> https://spam-classsifier.onrender.com/

---

## 📸 Preview

<img width="1539" height="1129" alt="image" src="https://github.com/user-attachments/assets/da224e65-4568-4581-9fb8-d2a02aa9cc3b" />

---

# Features

* AI-powered spam detection
* Custom NLP model trained with spaCy
* Real-time message classification
* Spam confidence score
* Spam / Authentic / Uncertain prediction
* Explanation for every prediction
* Clean and responsive user interface
* REST API backend
* Ready for deployment

---

# Tech Stack

### Backend

* Python
* Flask
* spaCy
* Pandas

### Frontend

* HTML5
* CSS3
* JavaScript

### Deployment

* GitHub
* Render

---

# Project Structure

```
SpamClasssifier_Website/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── fake_review_model/
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    ├── script.js
    └── assets/
```

---

# How It Works

1. A user enters a message.
2. The frontend sends the text to the Flask API.
3. The backend loads the trained spaCy model.
4. The model predicts the probability of spam.
5. The backend returns:

   * Label
   * Spam score
   * Confidence
   * Description
6. The frontend displays the prediction instantly.

---

Example Response

```json
{
    "label": "spam",
    "spam_score": 94,
    "description": "Multiple manipulation patterns detected.",
    "raw": {
        "spam_probability": 0.94,
        "authentic_probability": 0.06
    }
}
```

---

# Running Locally

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SpamClasssifier_Website.git
```

Move into the project

```bash
cd SpamClasssifier_Website
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# Machine Learning Model

The classifier is trained using a custom **spaCy TextCategorizer**.

Training process:

* Dataset preprocessing
* Duplicate removal
* Binary label encoding
* Training with spaCy
* Model serialization
* Local inference using `spacy.load()`

The trained model is stored in:

```
fake_review_model/
```

---

# Author

**Gourav Dutta**

Computer Science Undergraduate
Aspiring Machine Learning Engineer

GitHub:

```
https://github.com/yupitsmegd7
```

---

# License

This project is released under the MIT License.
