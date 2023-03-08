# ----- Load base libraries and packages
import gradio as gr

import numpy as np
import pandas as pd

import pickle

import xgboost as xgb
from xgboost import XGBClassifier

# ----- Useful lists
expected_inputs = ["tenure", "montant", "frequence_rech", "arpu_segment", "frequence", "data_volume", "regularity", "freq_top_pack"]
columns_to_scale = ["montant", "frequence_rech", "arpu_segment", "frequence", "data_volume", "regularity", "freq_top_pack"]
categoricals = ["tenure"]


# ----- Helper Functions
# Function to load ML toolkit
def load_ml_toolkit(file_path=r"src\Gradio_App_toolkit"):
    """
    This function loads the ML items into this app by taking the path to the ML items.

    Args:
        file_path (regexp, optional): It receives the file path to the ML items, but defaults to the "src" folder in the repository.

    Returns:
        file: It returns the pickle file (which in this case contains the Machine Learning items.)
    """

    with open(file_path, "rb") as file:
        loaded_toolkit = pickle.load(file)
    return loaded_toolkit


# Importing the toolkit
loaded_toolkit = load_ml_toolkit(r"src\Gradio_App_toolkit")
encoder = loaded_toolkit["encoder"]
scaler = loaded_toolkit["scaler"]

# Import the model
model = XGBClassifier()
model.load_model(r"src\xgb_model.json")


# Function to process inputs and return prediction
def process_and_predict(*args, encoder=encoder, scaler=scaler, model=model):
    """
    This function processes the inputs and returns the predicted churn status of the customer.
    It receives the user inputs, the encoder, scaler and model. The inputs are then put through the same process as was done during modelling, including but not limited to encoding of categorical columns and scaling of numeric columns.

    Args:
        encoder (LabelEncoder, optional): It is the encoder used to encode the categorical features before training the model, and should be loaded either as part of the ML items or as a standalone item. Defaults to encoder, which comes with the ML Items dictionary.
        scaler (MinMaxScaler, optional): It is the scaler (MinMaxScaler) used to scale the numeric features before training the model, and should be loaded either as part of the ML Items or as a standalone item. Defaults to scaler, which comes with the ML Items dictionary.
        model (XGBoost, optional): This is the model that was trained and is to be used for the prediction. It defaults to "model", as loaded from the ML toolkit.

    Returns:
        Prediction (label): Returns the label of the predicted class, i.e. one of whether the given customer will churn or not.
    """

    # Convert inputs into a DataFrame
    input_data = pd.DataFrame([args], columns=expected_inputs)

    # Encode the categorical column
    input_data["tenure"] = encoder.transform(input_data["tenure"])
    
    # Scale the numeric columns
    input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])

    # Making the prediction
    model_output = model.predict(input_data)
    return {"Prediction: CHURN": float(model_output[0]), "Prediction: STAY": 1-float(model_output[0])}


# ----- App Interface
with gr.Blocks() as turn_on_the_gradio:
    gr.Markdown("# Expresso Customer Churn Prediction")
    gr.Markdown("""This app uses a machine learning model to predict whether or not a customer will churn based on inputs made by you, the user. The (XGBoost) model was trained and built based on Zindi's Expresso Dataset. You may refer to the expander at the bottom for more information on the inputs.""")
    
    # Phase 1: Receiving Inputs
    # Usage
    gr.Markdown("**Customer Usage Data**")
    with gr.Row():
        montant = gr.Slider(label="Top-up amount", minimum=20, step=1, interactive=True, value=1, maximum= 500000)
        data_volume = gr.Slider(label="Number of connections", minimum=0, step=1, interactive=True, value=1, maximum= 2000000)

    # Activity
    gr.Markdown("**Activity Levels**")
    with gr.Row():
        frequence_rech = gr.Slider(label="Recharge Frequency", minimum=1, step=1, interactive=True, value=1, maximum=220)
        freq_top_pack = gr.Slider(label="Top Package Activation Frequency", minimum=1, step=1, interactive=True, value=1, maximum=1050)
        regularity = gr.Slider(label="Days of Activity (out of 90 days)", minimum=1, step=1, interactive=True, value=1, maximum=90)        
        tenure = gr.Dropdown(label="Tenure (time on the network)", choices=["D 3-6 month", "E 6-9 month", "F 9-12 month", "G 12-15 month", "H 15-18 month", "I 18-21 month", "J 21-24 month", "K > 24 month"], value="K > 24 month")

    # Income
    gr.Markdown("**Income from the Customer**")
    with gr.Row():
        arpu_segment = gr.Slider(label="Income over the last 90 days", step=1, maximum=287000, interactive=True)
        frequence = gr.Slider(label="Number of times the customer has made an income", step=1, minimum=1, maximum=91, interactive=True)

    # Output Prediction
    output = gr.Label("Awaiting Submission...")
    submit_button = gr.Button("Submit")
    
    # Expander for more info on columns
    with gr.Accordion("Open for information on inputs"):
        gr.Markdown("""This app receives the following as inputs and processes them to return the prediction on whether a customer, given the inputs, will churn or not.
                    - arpu_segment: income over 90 days / 3
                    - churn: variable to predict - target
                    - data_volume: number of connections
                    - frequence: number of times the client has made an income
                    - frequence_rech: number of times the client recharged
                    - freq_top_pack: number of times the client has activated the top pack packages
                    - montant: top-up amount
                    - regularity: number of times the client is active for 90 days
                    - tenure: duration in the network
                    """)
    
    submit_button.click(fn = process_and_predict,
                        outputs = output,
                        inputs=[tenure, montant, frequence_rech, arpu_segment, frequence, data_volume, regularity, freq_top_pack])

turn_on_the_gradio.launch(favicon_path= r"src\app_thumbnail.png", inbrowser= True, share= True)