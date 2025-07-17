
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class WeatherPrediction:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.data = self.data.dropna()
        self.model = None

    def train(self):
        X = self.data[['Humidity', 'Pressure (millibars)', 'Wind Speed (km/h)']]
        y = self.data['Temperature (C)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

    def predict(self, new):
        if self.model is None:
            raise Exception("Model not trained. Call train() first.")
        prediction = self.model.predict([new])
        return prediction[0]
