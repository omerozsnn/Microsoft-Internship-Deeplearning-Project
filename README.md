
# Stock Price Prediction for AAPL (Apple)

This project is designed to predict the stock price of Apple Inc. (AAPL) using deep learning models built with PyTorch. The model leverages historical stock price data and aims to provide accurate forecasts of future stock prices.

## Project Overview

This project was developed as part of the FY24 Microsoft team project. The goal is to implement a stock price prediction model that utilizes historical data and predicts future stock prices through machine learning and deep learning techniques.

### Key Components:
- **Data Preprocessing:** Data is cleaned, normalized, and split into training and test sets using `Pandas` and `scikit-learn`.
- **Modeling:** A deep learning model using PyTorch's `nn.Module` is implemented to perform the time series prediction.
- **Visualization:** The results are visualized using `Matplotlib` and `Seaborn` to show the actual vs predicted prices.
- **Device Agnostic:** The code is optimized to run on GPU if available, using PyTorch's `cuda` functionality.

## Getting Started

### Prerequisites
To run this project, you need to install the following dependencies:

```bash
pip install pandas numpy scikit-learn torch matplotlib seaborn
```

Additionally, if you are running the project in Google Colab, you can connect your Google Drive for data storage and access.

### Running the Project
1. **Data Preparation:**
   Ensure you have the stock price data for AAPL, which can be uploaded to your Colab environment or loaded from your Google Drive.
   
2. **Execute the Jupyter Notebook:**
   Open the `stock_price_predictions_AAPL.ipynb` notebook in Google Colab or Jupyter, and run each cell step by step.

3. **Model Training:**
   The model will be trained on historical stock price data. You can modify the hyperparameters such as learning rate, batch size, or the number of epochs in the notebook to fine-tune the model.

4. **Prediction and Visualization:**
   After training, the model will predict the stock prices and generate visualizations comparing the actual vs predicted values.

## Project Structure

- **Importing Necessary Libraries:** Imports all the libraries and packages required for the project.
- **Data Preprocessing:** Loads the dataset and performs the necessary data cleaning and normalization.
- **Model Creation:** Defines the neural network model using PyTorch.
- **Training:** Trains the model on the preprocessed data.
- **Evaluation:** Evaluates the model performance and generates visualizations.

## Author
This project was created by Ömer Özsan as part of a FY24 Microsoft team project.

## License
This project is licensed under the MIT License.
