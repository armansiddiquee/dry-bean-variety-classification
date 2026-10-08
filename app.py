import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    """
    Load the trained Random Forest model.
    """
    return joblib.load("models/random_forest_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(
        "Unable to load the Random Forest model. "
        "Please make sure 'models/random_forest_model.pkl' "
        "exists in the correct folder."
    )
    st.stop()


# =========================================================
# FEATURE NAMES
# =========================================================

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]


# =========================================================
# APPLICATION HEADER
# =========================================================

st.title("🌱 Dry Bean Classification")

st.markdown(
    """
    ### Machine Learning Classification Application

    This application predicts the **class of a dry bean**
    using 16 morphological features from the **UCI Dry Bean Dataset**.

    The application uses the **Random Forest Classifier** selected
    from the machine learning experiments performed in the
    accompanying Jupyter Notebook.
    """
)

st.info(
    "Deployment Model: Random Forest Classifier | "
    "Test Accuracy: approximately 92.03%"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("About the Model")

st.sidebar.write(
    """
    **Model:** Random Forest Classifier

    **Dataset:** UCI Dry Bean Dataset

    **Input Features:** 16

    **Number of Classes:** 7

    **Test Accuracy:** 92.03%
    """
)

st.sidebar.divider()

st.sidebar.write(
    """
    **Application workflow**

    1. Enter bean measurements.
    2. Click Predict.
    3. The trained model predicts the bean class.

    You can also upload the provided `test_data.csv`
    to evaluate the deployed model.
    """
)


# =========================================================
# SINGLE RECORD PREDICTION
# =========================================================

st.header("🔬 Predict a Dry Bean Class")

st.write(
    "Enter the 16 morphological measurements of a dry bean."
)

st.warning(
    "For meaningful predictions, enter values within the "
    "range represented in the Dry Bean dataset."
)


# =========================================================
# INPUT FORM
# =========================================================

with st.form("prediction_form"):

    st.subheader("Bean Morphological Features")

    # First row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        area = st.number_input(
            "Area",
            min_value=0.0,
            value=10000.0,
            format="%.6f"
        )

    with col2:
        perimeter = st.number_input(
            "Perimeter",
            min_value=0.0,
            value=400.0,
            format="%.6f"
        )

    with col3:
        major_axis_length = st.number_input(
            "MajorAxisLength",
            min_value=0.0,
            value=150.0,
            format="%.6f"
        )

    with col4:
        minor_axis_length = st.number_input(
            "MinorAxisLength",
            min_value=0.0,
            value=100.0,
            format="%.6f"
        )

    # Second row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        aspect_ration = st.number_input(
            "AspectRation",
            min_value=0.0,
            value=1.5,
            format="%.6f"
        )

    with col6:
        eccentricity = st.number_input(
            "Eccentricity",
            min_value=0.0,
            value=0.7,
            format="%.6f"
        )

    with col7:
        convex_area = st.number_input(
            "ConvexArea",
            min_value=0.0,
            value=10500.0,
            format="%.6f"
        )

    with col8:
        equiv_diameter = st.number_input(
            "EquivDiameter",
            min_value=0.0,
            value=115.0,
            format="%.6f"
        )

    # Third row
    col9, col10, col11, col12 = st.columns(4)

    with col9:
        extent = st.number_input(
            "Extent",
            min_value=0.0,
            value=0.75,
            format="%.6f"
        )

    with col10:
        solidity = st.number_input(
            "Solidity",
            min_value=0.0,
            value=0.98,
            format="%.6f"
        )

    with col11:
        roundness = st.number_input(
            "roundness",
            min_value=0.0,
            value=0.8,
            format="%.6f"
        )

    with col12:
        compactness = st.number_input(
            "Compactness",
            min_value=0.0,
            value=0.7,
            format="%.6f"
        )

    # Fourth row
    col13, col14, col15, col16 = st.columns(4)

    with col13:
        shape_factor1 = st.number_input(
            "ShapeFactor1",
            min_value=0.0,
            value=0.006,
            format="%.8f"
        )

    with col14:
        shape_factor2 = st.number_input(
            "ShapeFactor2",
            min_value=0.0,
            value=0.002,
            format="%.8f"
        )

    with col15:
        shape_factor3 = st.number_input(
            "ShapeFactor3",
            min_value=0.0,
            value=0.5,
            format="%.8f"
        )

    with col16:
        shape_factor4 = st.number_input(
            "ShapeFactor4",
            min_value=0.0,
            value=0.99,
            format="%.8f"
        )

    st.divider()

    predict_button = st.form_submit_button(
        "🌱 Predict Bean Class",
        type="primary",
        use_container_width=True
    )


# =========================================================
# MAKE SINGLE PREDICTION
# =========================================================

if predict_button:

    # Create DataFrame using the exact feature order
    input_data = pd.DataFrame(
        [[
            area,
            perimeter,
            major_axis_length,
            minor_axis_length,
            aspect_ration,
            eccentricity,
            convex_area,
            equiv_diameter,
            extent,
            solidity,
            roundness,
            compactness,
            shape_factor1,
            shape_factor2,
            shape_factor3,
            shape_factor4
        ]],
        columns=FEATURES
    )

    try:

        # Make prediction
        prediction = model.predict(input_data)[0]

        st.success(
            f"🌱 Predicted Dry Bean Class: **{prediction}**"
        )

        # Display entered values
        with st.expander("View Input Features"):

            st.dataframe(
                input_data,
                use_container_width=True
            )

        # Display prediction probabilities if available
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            classes = model.classes_

            probability_df = pd.DataFrame(
                {
                    "Class": classes,
                    "Probability": probabilities
                }
            )

            probability_df = probability_df.sort_values(
                by="Probability",
                ascending=False
            )

            st.subheader("Prediction Probabilities")

            st.dataframe(
                probability_df,
                use_container_width=True
            )

            st.bar_chart(
                probability_df.set_index("Class")["Probability"]
            )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# =========================================================
# CSV TESTING SECTION
# =========================================================

st.divider()

st.header("📂 Test the Model Using CSV")

st.write(
    """
    Upload the `test_data.csv` generated from the same test set
    used during model evaluation.
    
    The CSV should contain the 16 feature columns and may also
    contain the `Class` column containing the actual target.
    """
)


uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        uploaded_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.write(
            f"Number of records: **{len(uploaded_data)}**"
        )

        st.write(
            f"Number of columns: **{len(uploaded_data.columns)}**"
        )

        # -------------------------------------------------
        # Check required features
        # -------------------------------------------------

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in uploaded_data.columns
        ]

        if missing_features:

            st.error(
                "The uploaded CSV is missing the following "
                f"required features: {missing_features}"
            )

        else:

            # -------------------------------------------------
            # Make predictions
            # -------------------------------------------------

            predictions = model.predict(
                uploaded_data[FEATURES]
            )

            result_data = uploaded_data.copy()

            result_data["Predicted_Class"] = predictions

            # -------------------------------------------------
            # Display results
            # -------------------------------------------------

            st.subheader("Prediction Results")

            st.dataframe(
                result_data,
                use_container_width=True
            )

            # -------------------------------------------------
            # Calculate accuracy if actual Class exists
            # -------------------------------------------------

            if "Class" in uploaded_data.columns:

                accuracy = (
                    uploaded_data["Class"]
                    == result_data["Predicted_Class"]
                ).mean()

                st.subheader("Model Performance on Uploaded Data")

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:

                    st.metric(
                        "Prediction Accuracy",
                        f"{accuracy * 100:.2f}%"
                    )

                with metric_col2:

                    correct_predictions = (
                        uploaded_data["Class"]
                        == result_data["Predicted_Class"]
                    ).sum()

                    st.metric(
                        "Correct Predictions",
                        f"{correct_predictions:,}"
                    )

                # -------------------------------------------------
                # Performance interpretation
                # -------------------------------------------------

                if accuracy >= 0.90:

                    st.success(
                        "The deployed model is performing strongly "
                        "on the uploaded test data."
                    )

                elif accuracy >= 0.80:

                    st.info(
                        "The deployed model shows good performance "
                        "on the uploaded test data."
                    )

                else:

                    st.warning(
                        "The prediction accuracy is lower than "
                        "expected. Please verify the model and "
                        "feature columns."
                    )

            # -------------------------------------------------
            # Download predictions
            # -------------------------------------------------

            csv_output = result_data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv_output,
                file_name="dry_bean_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded CSV: {e}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Dry Bean Classification | Machine Learning Assignment"
)