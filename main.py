
from classifier.weather_prediction import WeatherPrediction

def main():
    prediction = WeatherPrediction("data/weatherHistory.csv")
    prediction.train()
    print("Temprature", prediction.predict([0.90, 1015.13, 14.1197]))

if __name__ == "__main__":
    main()
