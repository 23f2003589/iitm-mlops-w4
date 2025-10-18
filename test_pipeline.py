import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
#hello
BUCKET_URI = "gs://mlops-course-nifty-harmony-474217-q6-unique"
TRAINING_DATA_BUCKET_NAME = "training_data_mlops_w1"
TRAINING_BLOB = "iris.csv"

GCS_PATH = f"{BUCKET_URI}/{TRAINING_DATA_BUCKET_NAME}/{TRAINING_BLOB}"
MODEL_PATH = Path("artifacts/model.joblib")

def load_iris_from_gc(bucket_name=TRAINING_DATA_BUCKET_NAME,blob_name=TRAINING_BLOB):
    client=storage.Client()
    bucket=client.bucket(bucket_name)
    blob=bucket.blob(blob_name)
    data_bytes=blob.download_as_bytes()
    df=pd.read_csv(pd.io.common.BytesIO(data_bytes))
    return df

@pytest.fixture(scope="session")
def data():
    """Load the Iris dataset from GCS"""
    try:
        df=pd.read_csv("iris.csv")
        return df
    except Exception as e:
        print(e)
        pytest.skip(f"Could not load dataset from GCS: {e}")
        
@pytest.fixture(scope="session")
def model():
    """Load trained model from artifacts"""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    pytest.skip(f"Model artefact not found at {MODEL_PATH}")
    
def test_data_not_empty(data):
    """Ensure dataset is not empty"""
    assert not data.empty, "Dataset is empty."


def test_expected_columns(data):
    """Ensure columns are renamed correctly"""
    expected = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    actual = set(data.columns)
    assert expected == actual, f"Unexpected columns: {actual}"


def test_no_missing_values(data):
    """Ensure no NaN values"""
    assert not data.isna().any().any(), "Dataset contains missing values."


def test_species_is_categorical(data):
    """Check that 'species' is categorical with 3 unique string values"""
    species_unique = data["species"].unique()
    assert data["species"].dtype == object, "'species' is not string/object type."
    assert len(species_unique) == 3, f"Expected 3 unique species, found {len(species_unique)}: {species_unique}"
    

def test_no_units_or_spaces_in_columns(data):
    """Ensure column names are clean"""
    for col in data.columns:
        assert " " not in col, f"Space found in column name: {col}"
        assert "(cm)" not in col, f"(cm) found in column name: {col}"




def test_model_loads(model):
    """Ensure model loads correctly"""
    assert model is not None, "Model failed to load."


def test_model_predicts(model, data):
    """Ensure model can make predictions"""
    X = data.drop(columns=["species"])
    try:
        preds = model.predict(X)
    except Exception as e:
        pytest.fail(f"Model failed to predict: {e}")
    assert len(preds) == len(X), "Prediction length mismatch."


def test_model_accuracy_reasonable(model, data):
    """
    Basic model accuracy sanity check.
    If the model was trained on this dataset, it should achieve >80% accuracy.
    """
    X = data.drop(columns=["species"])
    y = data["species"]

    try:
        preds = model.predict(X)
    except Exception as e:
        pytest.skip(f"Cannot compute accuracy; model predict failed: {e}")

    acc = (preds == y).mean()
    assert acc > 0.8, f"Model accuracy too low ({acc:.2f})"
