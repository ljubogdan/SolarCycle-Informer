# Sunspots Prediction using Machine Learning

## Project Overview
This project leverages advanced machine learning techniques to predict sunspot activity using time series data. The focus is on implementing an LSTM (Long Short-Term Memory) model for sequential data processing, optimizing model performance through finding the optimal learning rate, and using Huber loss for robustness against outliers.

## Technologies Used
- **Python**: Core programming language for implementation.
- **TensorFlow**: Framework for building and training the LSTM model.
- **Pandas**: Data handling and manipulation in CSV format.
- **NumPy**: Numerical data processing and matrix operations.
- **Matplotlib & Seaborn**: Data and result visualization (plots, boxplots, distributions).
- **Keras**: API for defining and training neural networks.
- **Scikit-learn**: Metrics for model performance evaluation (MAE).
- **Jupyter Notebook/Google Colab**: Interactive environment for code development and testing.

## Features
- **Data Loading**: Analysis and preprocessing of the sunspots dataset from Kaggle.
- **Data Visualization**: Visual exploration of time series data, including full cycles and statistical distributions.
- **LSTM Model**: Implementation of a sequential model with Conv1D and LSTM layers for time series forecasting.
- **Huber Loss**: Robust loss function to handle outliers in the dataset.
- **Learning Rate Optimization**: Dynamic learning rate scheduling to identify the optimal learning rate.
- **Model Evaluation**: Mean Absolute Error (MAE) used to assess prediction accuracy on validation data.
- **Forecasting**: Sliding window approach for generating predictions based on historical data.

## Dataset
The dataset used is the [Sunspots dataset](https://www.kaggle.com/datasets/robervalt/sunspots), containing monthly mean total sunspot numbers from January 1749 to January 2021. It includes 3265 data points with no missing values, covering solar cycles from 1749 to the start of the 25th cycle in 2019.

## Model Architecture
- **Input Layer**: Conv1D with 132 filters and ReLU activation for smoothing time series data.
- **LSTM Layers**: Two LSTM layers (256 and 132 units) for capturing long-term dependencies.
- **Dense Layers**: Fully connected layers (80 and 10 units) with ReLU activation, followed by a single output unit.
- **Output Scaling**: Lambda layer to scale predictions to the range of original data.
- **Loss Function**: Huber loss with delta=1 for robust regression.
- **Optimizer**: SGD with momentum (0.9) and dynamically tuned learning rate.

## Key Parameters
- **Window Size**: 60 months for sequence generation.
- **Batch Size**: 145 for training data.
- **Shuffle Buffer**: 900 for randomizing dataset.
- **Epochs**: 100 for learning rate optimization, 200 for final training.
- **Split**: 90% training, 10% validation.

## Results
- **Optimal Learning Rate**: Determined through a learning rate scheduler and visualized via loss vs. learning rate plot.
- **Validation MAE**: Mean Absolute Error calculated to evaluate model performance on unseen data.
- **Visualizations**: Plots comparing actual validation data with predicted values, alongside MAE and Huber loss trends during training.

## How to Run
- **Install Dependencies**:
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn
- **Download Dataset**: Import the Sunspots dataset from [Kaggle](https://www.kaggle.com/datasets/robervalt/sunspots) or SILSO database (https://sidc.be)
- **Run the Notebook**: Use Jupyter Notebook or Google Colab to execute the provided code.
- **Explore Results**: Review generated plots and MAE metrics for model performance.

## Future Improvements
- Experiment with different window sizes or batch sizes to improve prediction accuracy.
- Incorporate additional features like solar cycle metadata.
- Test alternative models like GRU or Transformer-based architectures.
- Adjust Huber loss delta to further optimize for outliers.

## References
- [Sunspots Dataset](https://www.kaggle.com/datasets/robervalt/sunspots)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Huber Loss](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Huber)
- [LSTM Explanation](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## License
This project is licensed under the MIT License.