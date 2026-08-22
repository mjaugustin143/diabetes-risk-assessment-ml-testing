import os
import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

def evaluate_patient(pregnancies, glucose, blood_pressure, skin_thickness, insulin, weight, height, pedigree_choice, age):
    try:
        # Calculate BMI automatically
        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 1)
        
        # Map friendly family history choices to numerical scores
        pedigree_map = {
            "No direct family history": 0.2,
            "One relative (moderate risk)": 0.5,
            "Multiple close relatives (high risk)": 1.1
        }
        pedigree = pedigree_map[pedigree_choice]

        # Prepare input DataFrame with exact training column names
        input_df = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, pedigree, age
        ]], columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])

        # Scale features and predict
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]

        # Format output
        if prediction == 1:
            result = "🔴 High Diabetes Risk Detected"
            confidence = f"{probabilities[1] * 100:.1f}% Risk Probability"
        else:
            result = "🟢 Low Diabetes Risk Detected"
            confidence = f"{probabilities[0] * 100:.1f}% Safety Confidence"

        return result, confidence, f"{bmi} kg/m²"

    except Exception as e:
        return f"Error: {str(e)}", "N/A", "N/A"

# Build UI without passing theme here (prevents deprecation warning)
with gr.Blocks(title="Diabetes Risk Assessment") as demo:
    gr.Markdown("# 🩺 Diabetes Risk Assessment Tool")
    gr.Markdown("Fill in the patient details below to evaluate the likelihood of diabetes.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Personal Information & Metrics")
            age = gr.Number(label="Age (years)", value=30, precision=0)
            pregnancies = gr.Number(label="Pregnancies", value=1, precision=0)
            weight = gr.Number(label="Weight (kg)", value=70.0)
            height = gr.Number(label="Height (cm)", value=165.0)
            
            gr.Markdown("### 🧬 Family History")
            pedigree_choice = gr.Radio(
                choices=["No direct family history", "One relative (moderate risk)", "Multiple close relatives (high risk)"],
                value="No direct family history",
                label="Family History of Diabetes"
            )

        with gr.Column():
            gr.Markdown("### 🩸 Clinical Measurements")
            glucose = gr.Slider(label="Glucose Level (mg/dL)", minimum=0, maximum=300, value=120)
            blood_pressure = gr.Slider(label="Blood Pressure (mm Hg)", minimum=0, maximum=180, value=70)
            skin_thickness = gr.Slider(label="Skin Fold Thickness (mm)", minimum=0, maximum=100, value=20)
            insulin = gr.Slider(label="Insulin Level (mu U/ml)", minimum=0, maximum=900, value=80)

    btn = gr.Button("🔍 Evaluate Diabetes Risk", variant="primary")
    
    gr.Markdown("---")
    gr.Markdown("## 📊 Assessment Result")
    
    with gr.Row():
        result_box = gr.Textbox(label="Assessment Result")
        confidence_box = gr.Textbox(label="Model Confidence")
        bmi_box = gr.Textbox(label="Calculated BMI")

    btn.click(
        fn=evaluate_patient,
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, weight, height, pedigree_choice, age],
        outputs=[result_box, confidence_box, bmi_box]
    )

# Adapted launch settings for Render compatibility
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft()
    )
