import cv2
import os
import mahotas
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

model_path = os.path.join("model_v2.keras")
scaler_path = os.path.join("scaler_v2.save")

# Extract Features
fixed_size = (300, 300)
bins = 8

class_names = ["Bacterialblight", "Brown Spot", "Healthy", "LeafBlast"]

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(0)
    return haralick

def fd_histogram(image, bins=8):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def pre_process(image):
    pass