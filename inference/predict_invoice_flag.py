import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_COLUMNS = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

scaler = joblib.load(SCALER_PATH)

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained Classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data: dict

    Returns
    -------
    pd.DataFrame with predicted invoice risk outputs
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)

    missing_columns = [column for column in FEATURE_COLUMNS if column not in input_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required input columns: {missing_columns}. "
            f"Expected columns: {FEATURE_COLUMNS}"
        )

    feature_frame = input_df[FEATURE_COLUMNS]
    input_scaled = scaler.transform(feature_frame)
    flag_probability = model.predict_proba(input_scaled)[:, 1]
    prediction = (flag_probability >= 0.5).astype(int)

    input_df["Predicted_Flag"] = prediction
    input_df["Flag_Probability"] = flag_probability.round(4)

    return input_df

if __name__ == "__main__":
    # Example inference run (local testing)

    sample_data = {
        "invoice_quantity": [4, 2, 1, 1],
        "invoice_dollars": [18500, 9000, 3000, 200],
        "Freight": [1.73, 1.73, 1.73, 1.73],
        "total_item_quantity": [4, 2, 1, 1],
        "total_item_dollars": [18500, 9000, 3000, 200],
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)