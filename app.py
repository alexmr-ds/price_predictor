"""
Streamlit app for interactive car price prediction with SHAP explanations.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Add src to path
PROJ_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_utils import inference, cleaning_utils


st.set_page_config(page_title="Car Price Predictor", layout="wide")


def range_imputer(val_dict, feature):
    """
    Returns the values of the range for the specificed feature column
    """
    dict = val_dict[feature]
    min = dict.get("min")
    max = dict.get("max")
    median = dict.get("default")

    return {"min_value": min, "max_value": max, "value": median}


def collect_user_input(brand_models: dict):
    """
    Collect vehicle characteristics from the user via a Streamlit sidebar interface.

    This function dynamically constructs user input widgets based on brand–model–
    specific feature ranges and categorical availability derived from the training
    dataset. Numerical input bounds are constrained using precomputed, data-driven
    ranges to ensure realistic and model-consistent values, while categorical
    selections are limited to options observed for the selected brand–model
    combination.

    When available, brand–model–specific ranges are used; otherwise, global
    fallback ranges are applied.

    Parameters
    ----------
    brand_models: dict
        Dictionary mapping each vehicle brand to a list of available models.
        Used to populate the brand and model selection widgets.
    brand_model_ranges: dict
        Dictionary with numerical brand-model grouping inputs
    cat_cols_option: dict
        Dictionary with the available categorical options per brand-model grouping

    Returns
    -------
    dict
        Dictionary containing the user-provided vehicle attributes formatted
        according to the raw schema expected by the preprocessing and prediction
        pipeline. The returned keys include both numerical and categorical features
        required for inference.
    """

    st.sidebar.header("Vehicle details")

    brand = st.sidebar.selectbox("Brand", sorted(brand_models.keys()))
    model_options = brand_models.get(brand, [])
    model = (
        st.sidebar.selectbox("Model", model_options)
        if model_options
        else st.sidebar.text_input("Model")
    )

    # Import dicts of brand-model grouping inputs
    brand_model_ranges, cat_cols_options = inference.build_brand_model_ranges()
    selected_ranges = brand_model_ranges.get((brand, model))
    selected_categories = cat_cols_options.get((brand, model), {})

    # --- Feature inputs driven by brand–model ranges --- #
    year = st.sidebar.number_input(
        "year", **range_imputer(selected_ranges, "year"), step=1
    )

    mileage = st.sidebar.slider(
        "Mileage (miles)",
        **range_imputer(selected_ranges, "mileage"),
    )

    engine_size = st.sidebar.number_input(
        "Engine size (L)",
        step=1,
        **range_imputer(selected_ranges, "engineSize"),
    )

    transmission = st.sidebar.selectbox(
        "Transmission",
        [
            category
            for category in selected_categories["transmission"]
            if category.lower() != "unknown"
        ],
        index=0,
    )
    mpg = 0

    return {
        "Brand": brand,
        "model": model,
        "year": year,
        "transmission": transmission,
        "mileage": mileage,
        "engineSize": engine_size,
        "mpg": mpg,
    }


def main():
    st.title("Car Price Prediction")
    st.write(
        "Fill in the vehicle details in the sidebar and click Predict to get a price "
        "estimate."
    )

    model = inference.load_final_model()

    brand_models = inference.load_brand_model_mapping()

    payload = collect_user_input(brand_models)

    if st.button("Predict"):
        raw_df = inference.build_raw_input(payload)
        feature_engineered_df = cleaning_utils.engineer_features(
            raw_df, log_features=["mileage", "mpg"]
        )

        # Stacking model expects raw input; its base estimators handle preprocessing
        price = inference.predict_price(model, feature_engineered_df)

        # st.success(f"Predicted price: £{price:,.0f}")
        st.markdown(
            f"""
            <div style="text-align:center; font-size:32px; font-weight:600;">
                Estimated price:<br>
                <span style="font-size:40px;">£{price:,.0f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
