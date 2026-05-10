# Ai-Ml-task-2
his repository features three advanced AI/ML implementations:  A robust XGBoost pipeline for customer churn prediction.  A multimodal neural network (CNN+MLP) in TensorFlow for price forecasting.  A context-aware RAG chatbot using LangChain and FAISS. It demonstrates expertise in classical ML, deep learning, and LLM architectures.
# Advanced AI & Machine Learning Portfolio

## Overview
This repository contains a comprehensive Jupyter Notebook (`AI_ML_TAKSsss_.ipynb`) demonstrating advanced proficiency across multiple domains of modern Artificial Intelligence and Machine Learning. The project is divided into three distinct, production-level tasks covering structured data modeling, multimodal deep learning, and Generative AI application development.

## Table of Contents
1. [Projects Included](#projects-included)
    * [Advanced ML Pipeline (XGBoost)](#1-advanced-ml-pipeline-customer-churn-prediction)
    * [Multimodal ML (TensorFlow/Keras)](#2-multimodal-ml-housing-price-prediction)
    * [Context-Aware Chatbot (RAG)](#3-context-aware-chatbot-rag)
2. [Technologies Used](#technologies-used)
3. [Installation & Setup](#installation--setup)
4. [Usage](#usage)

---

## Projects Included

### 1. Advanced ML Pipeline: Customer Churn Prediction
A production-grade machine learning workflow using the Telco Customer Churn dataset.
* **Architecture:** Utilizes Scikit-Learn's `Pipeline` and `ColumnTransformer` for robust data preprocessing (median imputation, standardization, and ordinal encoding).
* **Model:** Gradient Boosting via `XGBClassifier`.
* **Optimization:** Hyperparameter tuning implemented using `RandomizedSearchCV` to maximize the F1-score.
* **Output:** Includes a detailed classification evaluation and serializes the optimized model into a deployable artifact (`.pkl`).

### 2. Multimodal ML: Housing Price Prediction
A hybrid deep learning architecture designed to process heterogeneous data types simultaneously.
* **Architecture:** Built with TensorFlow and Keras. Features a dual-branch neural network:
  * **Branch A (Images):** A Convolutional Neural Network (CNN) extracts spatial features from synthetic house images (64x64x3).
  * **Branch B (Tabular):** A Multi-Layer Perceptron (MLP) processes numeric metadata (rooms, age, neighborhood score).
* **Fusion:** Concatenates the extracted features from both branches to predict a final continuous target variable (Housing Price).

### 3. Context-Aware Chatbot (RAG)
An intelligent, document-grounded conversational agent using Retrieval-Augmented Generation (RAG).
* **Framework:** Built using `LangChain` and `HuggingFaceEmbeddings`.
* **Vector Store:** Implements `FAISS` for efficient similarity search and context retrieval from internal knowledge bases.
* **Memory Management:** Features a custom manual memory buffer to maintain conversational context, allowing the AI to handle follow-up questions effectively without relying on unstable memory modules.

---

## Technologies Used
* **Languages:** Python 3
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deep Learning:** TensorFlow, Keras
* **NLP & GenAI:** LangChain, HuggingFace, FAISS, Sentence-Transformers
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib

---

## Installation & Setup

To run this notebook locally, ensure you have Python 3 installed. It is recommended to use a virtual environment.

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
