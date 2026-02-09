import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("mobile_price.pkl", "rb") as file:
    model = pickle.load(file)


def predict_price(
    battery_power,
    blue,
    clock_speed,
    dual_sim,
    fc,
    four_g,
    int_memory,
    m_dep,
    mobile_wt,
    n_cores,
    pc,
    px_height,
    px_width,
    ram,
    sc_h,
    sc_w,
    talk_time,
    three_g,
    touch_screen,
    wifi,
):

    input_data = pd.DataFrame(
        [
            [
                battery_power,
                blue,
                clock_speed,
                dual_sim,
                fc,
                four_g,
                int_memory,
                m_dep,
                mobile_wt,
                n_cores,
                pc,
                px_height,
                px_width,
                ram,
                sc_h,
                sc_w,
                talk_time,
                three_g,
                touch_screen,
                wifi,
            ]
        ],
        columns=[
            "battery_power",
            "blue",
            "clock_speed",
            "dual_sim",
            "fc",
            "four_g",
            "int_memory",
            "m_dep",
            "mobile_wt",
            "n_cores",
            "pc",
            "px_height",
            "px_width",
            "ram",
            "sc_h",
            "sc_w",
            "talk_time",
            "three_g",
            "touch_screen",
            "wifi",
        ],
    )

    prediction = model.predict(input_data)[0]

    return f"Predicted Price Range: {prediction}"


# [
#     "battery_power",
#     "blue",
#     "clock_speed",
#     "dual_sim",
#     "fc",
#     "four_g",
#     "int_memory",
#     "m_dep",
#     "mobile_wt",
#     "n_cores",
#     "pc",
#     "px_height",
#     "px_width",
#     "ram",
#     "sc_h",
#     "sc_w",
#     "talk_time",
#     "three_g",
#     "touch_screen",
#     "wifi",
# ]

inputs = [
    gr.Number(label="Battery Power"),
    gr.Radio([0, 1], label="Bluetooth"),
    gr.Number(label="Clock Speed"),
    gr.Radio([0, 1], label="Dual Sim"),
    gr.Number(label="Front Camera"),
    gr.Radio([0, 1], label="4G"),
    gr.Number(label="Internal Memory"),
    gr.Number(label="Mobile Depth"),
    gr.Number(label="Mobile Weight"),
    gr.Number(label="Number of Cores"),
    gr.Number(label="Primary Camera"),
    gr.Number(label="Pixel Height"),
    gr.Number(label="Pixel Width"),
    gr.Number(label="RAM"),
    gr.Number(label="Screen Height"),
    gr.Number(label="Screen Width"),
    gr.Number(label="Talk Time"),
    gr.Radio([0, 1], label="3G"),
    gr.Radio([0, 1], label="Touch Screen"),
    gr.Radio([0, 1], label="WiFi"),
]

app = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs="text",
    title="Mobile Price Range Prediction",
)

app.launch()