import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("mobile_price.pkl", "rb") as file:
  model = pickle.load(file)

