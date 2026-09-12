# 💎 Diamond Price Prediction System

## 📌 Project Overview

The Diamond Price Prediction System is a Machine Learning based web application developed using Python and Flask.

The system predicts the estimated market price of a diamond based on its physical characteristics and quality attributes.

The Machine Learning model used in this project is **Random Forest Regression**.

The application provides a user-friendly web interface where users can enter diamond specifications and receive an estimated diamond price in US dollars.

---

## 🎯 Objectives

The main objectives of this project are:

- To develop a Machine Learning model for diamond price prediction.
- To analyze the relationship between diamond characteristics and price.
- To use Random Forest Regression for price prediction.
- To develop a web-based prediction interface using Flask.
- To provide users with an easy-to-use diamond price prediction system.
- To integrate a trained Machine Learning model into a web application.

---

## ✨ Key Features

- 💎 Diamond price prediction
- 🤖 Random Forest Regression Machine Learning model
- 🌐 Flask-based web application
- 📊 Diamond dataset analysis
- 📝 User input form
- 💰 Estimated price displayed in USD
- 🎨 HTML and CSS based user interface
- 📱 Responsive web interface
- 📈 Model evaluation metrics
- ⚡ Fast prediction using a pre-trained model

---

## 🤖 Machine Learning Algorithm

### Random Forest Regression

Random Forest Regression is an ensemble Machine Learning algorithm that combines multiple decision trees to produce a more accurate prediction.

In this project, multiple decision trees are trained using diamond characteristics.

The predictions from these trees are combined to produce the final estimated diamond price.

### Why Random Forest?

Random Forest was selected because it:

- Handles nonlinear relationships effectively.
- Works well with numerical and categorical features.
- Reduces overfitting compared with a single decision tree.
- Provides good prediction performance.
- Can handle complex relationships between diamond characteristics and price.

---

## 💎 Diamond Features

The prediction system uses diamond characteristics such as:

- Carat
- Cut
- Color
- Clarity
- Depth
- Table
- X dimension
- Y dimension
- Z dimension

These characteristics are processed by the Machine Learning model to estimate the diamond's price.

---

## 🏗️ System Architecture

```text
                ┌─────────────────────┐
                │       User          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Web Interface     │
                │    HTML + CSS       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Flask Backend    │
                │       app.py        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Pre-trained ML      │
                │ Random Forest Model │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Predicted Diamond   │
                │       Price         │
                └─────────────────────┘