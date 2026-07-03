import gradio as gr
import joblib
import numpy as np

model = joblib.load('demand_forecast_model.pkl')
def predict_sales(Store, DayOfWeek,Promo,SchoolHoliday,CompetitionDisatnce,Year,Month,Day):
  features = np.array([[Store, DayOfWeek, Promo, 0, SchoolHoliday, 0, 0, CompetitionDisatnce, 0,Year,Month,Day,1,0,1,0,0]])
  prediction = model.predict(features)
  return f"Predicted Sales:dollar {round(prediction[0],2)}"

app = gr.Interface(
    fn = predict_sales,
    inputs = [
        gr.Number(label = "Store ID"),
        gr.Slider(1,7,step = 1,label = "Day of Week(1=Mon,7=Sun)"),
        gr.Radio([0,1],label = "Promo(0=No,1=yes)"),
        gr.Radio([0,1],label = "School Holiday(0=No,1=yes)"),
        gr.Number(label = "Competition Distance(meters)"),
        gr.Number(label = "Year(eg.2025)"),
        gr.Slider(1,12,step = 1,label = "Month"),
        gr.Slider(1,31,step=1,label = "Day"),
                  
                    
    ],
    outputs = gr.Textbox(label = "Prediction"),
    title = "Rossmann Sales Predictor",
    description = "Enter store details to predict daily sales!"
)

app.launch(share = True)