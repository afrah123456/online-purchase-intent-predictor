# Purchase Intent Prediction System

AI-powered system that predicts whether online shoppers will complete a purchase or abandon their cart.

**Live Demo:** https://purchase-intent-predictor.netlify.app  
**API:** https://purchase-intent-api.onrender.com

---

## Overview

This project uses machine learning to analyze customer browsing behavior and predict purchase intent in real-time. The system provides risk scores and actionable recommendations to help reduce cart abandonment.

**Key Results:**
- 86.5% prediction accuracy
- Sub-100ms response time
- 4-tier risk classification (Critical/High/Medium/Low)

---

## Tech Stack

**Backend:** FastAPI, Python, scikit-learn, XGBoost  
**Frontend:** React, Recharts  
**Models:** XGBoost, Random Forest, Logistic Regression

---

## Features

- Real-time purchase intent prediction
- Risk scoring with automated recommendations
- Live monitoring dashboard
- API performance tracking
- Prediction history
- Multiple ML models with ensemble option

---

## Quick Start

### Backend
```bash
cd backend/app
pip install -r ../requirements.txt
python main.py
```
Runs on `http://localhost:8000`

### Frontend
```bash
cd frontend
npm install
npm start
```
Runs on `http://localhost:3000`

---


Full API docs: https://purchase-intent-api.onrender.com/docs

---

## Model Performance

| Model | Accuracy | Recall |
|-------|----------|--------|
| XGBoost | 86.5% | 77.6% |
| Random Forest | 88.3% | 58.9% |
| Logistic Regression | 86.8% | 81.1% |

---

## Author

Afrah - [@afrah123456](https://github.com/afrah123456)

---

