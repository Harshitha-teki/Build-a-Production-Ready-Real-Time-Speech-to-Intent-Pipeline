"""Intent Classification module using TF-IDF + LogisticRegression."""

import os
import json
import joblib
import numpy as np
from typing import Tuple

INTENTS = [
    "TurnOn",
    "TurnOff",
    "SetBrightness",
    "GetTemperature",
    "PlayMusic",
    "StopMusic",
    "SetTimer",
    "GetWeather",
]

INTENT_RESPONSES = {
    "TurnOn": "Okay, turning it on.",
    "TurnOff": "Okay, turning it off.",
    "SetBrightness": "Okay, adjusting the brightness.",
    "GetTemperature": "The current temperature is 72 degrees Fahrenheit.",
    "PlayMusic": "Playing music now.",
    "StopMusic": "Stopping the music.",
    "SetTimer": "Timer has been set.",
    "GetWeather": "The weather is currently clear and sunny.",
}

TRAINING_DATA = {
    "TurnOn": [
        "turn on the lights", "turn on the light", "switch on the lamp",
        "turn on the fan", "turn on the ac", "turn on air conditioner",
        "power on the tv", "turn on the tv", "turn on television",
        "switch on the light", "turn on kitchen lights",
        "turn on the living room light", "turn on bedroom light",
        "enable the lights", "start the lights", "lights on please",
        "turn the light on", "switch the lights on", "power on the light",
        "can you turn on the lights", "please turn on the light",
        "activate the lights", "turn on all lights", "turn on hallway light",
        "turn on porch light", "turn on garage light", "turn on the heater",
        "turn on the fan please", "switch on the ac", "start the ac",
    ],
    "TurnOff": [
        "turn off the lights", "turn off the light", "switch off the lamp",
        "turn off the fan", "turn off the ac", "turn off air conditioner",
        "power off the tv", "turn off the tv", "turn off television",
        "switch off the light", "turn off kitchen lights",
        "turn off the living room light", "turn off bedroom light",
        "disable the lights", "stop the lights", "lights off please",
        "turn the light off", "switch the lights off", "power off the light",
        "can you turn off the lights", "please turn off the light",
        "deactivate the lights", "turn off all lights", "turn off hallway light",
        "turn off porch light", "turn off garage light", "turn off the heater",
        "turn off the fan please", "switch off the ac", "stop the ac",
    ],
    "SetBrightness": [
        "set brightness to 50 percent", "set the brightness to 75",
        "dim the lights to 30", "increase brightness to 80",
        "lower the brightness to 20", "set light level to 60",
        "adjust brightness to 40 percent", "make it brighter",
        "make it dimmer", "set brightness to maximum",
        "set brightness to minimum", "dim the living room lights to 50",
        "brighten the bedroom to 90", "set the lamp brightness to 70",
        "change brightness to 25", "set brightness level to 100",
        "increase light brightness", "decrease light brightness",
        "set the light intensity to 50", "adjust light brightness to 60",
        "can you dim the lights", "please set brightness to 45",
        "raise the brightness", "lower the brightness",
        "set the kitchen brightness to 80", "dim everything to 20",
        "brighten the room to 70", "set light to 50 percent",
    ],
    "GetTemperature": [
        "what is the temperature", "get the temperature",
        "how hot is it", "how cold is it", "what is the room temperature",
        "tell me the temperature", "check the temperature",
        "current temperature", "temperature reading",
        "what temperature is it outside", "how warm is it",
        "what is the indoor temperature", "read the thermostat",
        "what does the thermometer say", "get current temperature",
        "tell me how hot it is", "is it warm inside",
        "how is the temperature", "check room temperature",
        "what is the temperature now", "get me the temperature reading",
    ],
    "PlayMusic": [
        "play music", "play some music", "start playing music",
        "put on some music", "play a song", "play my playlist",
        "start the music", "play the radio", "turn on the music",
        "can you play music", "please play some music",
        "start playing a song", "play some tunes",
        "play background music", "play jazz", "play rock music",
        "play classical music", "play pop music",
        "shuffle my playlist", "play my favorite song",
        "start the playlist", "queue up some music",
        "play something", "play a tune",
    ],
    "StopMusic": [
        "stop the music", "stop playing music", "pause the music",
        "pause music", "turn off the music", "shut off the music",
        "halt the music", "end the music", "stop the song",
        "stop the radio", "pause the song", "stop playing",
        "can you stop the music", "please stop the music",
        "stop the playlist", "stop the tunes",
        "turn off the radio", "pause the radio",
        "stop playback", "stop the audio",
    ],
    "SetTimer": [
        "set a timer for 5 minutes", "set a timer for 10 minutes",
        "set a timer for 30 minutes", "set a timer for 1 hour",
        "start a timer for 15 minutes", "set timer for 20 minutes",
        "set a timer", "set an alarm for 7 am",
        "set a timer for 2 hours", "set a timer for 45 minutes",
        "create a timer for 10 minutes", "start a timer",
        "can you set a timer", "please set a timer for 5 minutes",
        "set a countdown for 30 minutes", "timer for 3 minutes",
        "remind me in 10 minutes", "set an alarm for tomorrow",
        "set a 5 minute timer", "set a 10 minute timer",
        "timer for one hour", "set timer 30 minutes",
        "set a timer for dinner", "start countdown 15 minutes",
    ],
    "GetWeather": [
        "what is the weather", "how is the weather",
        "get the weather", "tell me the weather",
        "weather forecast", "check the weather",
        "what is it like outside", "is it raining",
        "is it sunny", "is it cloudy",
        "what is the weather today", "how is the weather outside",
        "get me the weather forecast", "check weather outside",
        "is it going to rain", "will it rain today",
        "what is the temperature outside",
        "give me today's weather", "check the forecast",
        "what does the weather look like", "is the weather nice",
        "tell me about the weather", "current weather conditions",
        "weather report", "get today's forecast",
    ],
}


class IntentClassifier:
    """ML-based intent classifier using TF-IDF + LogisticRegression."""

    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

        if model_path and vectorizer_path and os.path.exists(model_path):
            self.load(model_path, vectorizer_path)
        elif model_path is None:
            self.train()

    def train(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        texts, labels = [], []
        for intent, phrases in TRAINING_DATA.items():
            for phrase in phrases:
                texts.append(phrase.lower())
                labels.append(intent)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), max_features=5000, sublinear_tf=True
        )
        X = self.vectorizer.fit_transform(texts)

        self.model = LogisticRegression(
            max_iter=1000, C=5.0, solver="lbfgs", multi_class="multinomial"
        )
        self.model.fit(X, labels)

    def save(self, model_path: str, vectorizer_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    def load(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str) -> Tuple[str, float]:
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        X = self.vectorizer.transform([text.lower()])
        intent = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        confidence = float(np.max(probs))
        return intent, confidence

    def get_response(self, intent: str) -> str:
        return INTENT_RESPONSES.get(intent, "I'm not sure how to help with that.")
