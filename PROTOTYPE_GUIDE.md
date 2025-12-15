# Beginner's Guide: Stock Prediction Prototype 📈

Welcome! This document will help you understand "what is going on under the hood" of your new Stock Predictor application. We've built this prototype using powerful tools but kept the concepts straightforward.

## 🚀 Quick Start
If you haven't already, you can start your application by running this command in your terminal:
```bash
streamlit run app.py
```
This launches a web server and opens the app in your browser.

---

## 🧩 How It Works (The Big Picture)
Imagine you are teaching a student to predict the weather. You give them a history of weather from the last 10 years. They look for patterns (e.g., "if it rains two days in a row in April, it stays sunny for the next week").

Your Stock Predictor works the same way:
1.  **Fetch Data**: It grabs the last few years of stock prices (like Apple or Google) from the internet.
2.  **Pre-process**: It turns those raw prices into a format the "brain" can understand (scaling numbers between 0 and 1).
3.  **Train (`train_model`)**: The "Brain" (an LSTM model) studies the past data. It guesses what comes next, sees if it was right, and corrects itself. It does this many times (Epochs).
4.  **Predict**: Once trained, it uses what it learned to draw a line for what it *thinks* happened or will happen.
5.  **Visualize**: We draw charts so you can see how well the AI's "guess" (Green line) matches the Real Price (Blue line).

---

## 🧠 Key Concepts Explained

### 1. The Model: LSTM (Long Short-Term Memory)
Standard AI models often just look at the current moment. But stock prices depend on *trends* over time. 
*   **Analogy**: If you read a sentence, you understand the last word because you remember the first word. 
*   **LSTM**: This is a special type of Neural Network designed to "remember" long-term trends (like a bull market) and short-term changes (a sudden drop yesterday). It is perfect for time-series data like stocks.

### 2. Epochs
*   **Definition**: One "Epoch" is one full cycle of reading through the entire history of data.
*   **Analogy**: Think of it as reading a textbook. If you read it once (1 Epoch), you might miss things. If you read it 50 times (50 Epochs), you understand it much better. However, reading it 1000 times might make you "memorize" it too much (Overfitting) and fail on new questions.

### 3. Sentiment Analysis 📰
Stocks aren't just moved by numbers; they are moved by news.
*   We use a tool (TextBlob) that reads recent news headlines about the company.
*   It assigns a "Polarity" score:
    *   **> 0 (Positive)**: Happy news (e.g., "Record Profits!")
    *   **< 0 (Negative)**: Sad news (e.g., "CEO Resigns")
    *   **0 (Neutral)**: Just facts.

---

## 📂 File Structure Walkthrough

Here is a map of your project files:

| File / Folder | Purpose |
| :--- | :--- |
| **`app.py`** | **The Interface**. This is the main file. It runs the website, creates the buttons, sliders, and charts you see in the browser. It coordinates everything else. |
| **`src/`** | **The Engine Room**. This folder contains all the logic. |
| `src/data_loader.py` | **The Fetcher**. Handles downloading data from Yahoo Finance and cleaning it up (removing errors, scaling numbers). |
| `src/model.py` | **The Brain**. Defines the structure of the Artificial Intelligence (the LSTM neural network). |
| `src/train.py` | **The Teacher**. Contains the loop that feeds data to the Brain and corrects it when it mistakes. |
| `src/strategy.py` | **The Trader**. A simple simulation script. It says: "If the AI predicts the price will go UP, buy. If DOWN, sell." and calculates how much money you'd make. |
| `requirements.txt` | **The Ingredients**. A list of all the Python libraries (like pandas, torch, streamlit) needed to run this recipe. |

---

## 🛠 Technologies Used
*   **Python**: The programming language.
*   **Streamlit**: Creates the beautiful web interface with zero HTML/CSS knowledge.
*   **PyTorch**: The heavy-duty Deep Learning library used to build the LSTM Brain.
*   **Pandas**: Used for handling tables of data (Excel for Python).
*   **Plotly**: Creates the interactive charts where you can zoom in and hover.

## 🔮 Next Steps for You
Now that you understand the prototype, here are some things you could try to learn locally:
1.  **Change the Ticker**: Try "GOOGL" (Google) or "TSLA" (Tesla) in the app sidebar.
2.  **Adjust Epochs**: Try increasing training epochs to 100. Does the Green line get closer to the Blue line?
3.  **Read the Code**: Open `app.py`. Try changing the title of the app (Line 12) and save. Refresh your browser to see the change!

Happy Coding! 🚀
