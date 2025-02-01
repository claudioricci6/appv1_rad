import streamlit as st
import joblib
import numpy as np
import sklearn

# Fix for compatibility issue
sklearn.tree._classes.ExtraTreeClassifier.monotonic_cst = None

def main():
    st.title("POPF Probability Calculator")

    # Initial message
    st.info("We recommend using the Pyradiomics software for feature extraction. Ensure to segment the pancreas to the left of the confluence between the splenic vein and the superior mesenteric vein.")

    # Load the joblib file
    try:
        model_data = joblib.load('/Users/claudio.ricci/Desktop/ML_Chat_GPT/app/appv1/app_rad/top5_rad_model.joblib')
        model = model_data['model']
        scaler = model_data['scaler']
        feature_labels = ['wLLH_FO_RMAD', 'wHLL_GLDM_SDHGLE', 'o_GLCM_CT', 'e_GLDM_SDLGLE', 'e_GLCM_SS']
    except Exception as e:
        st.error(f"Error loading the model: {e}. The model may not be compatible with the current scikit-learn version.")
        st.info("Please check the scikit-learn version or regenerate the model.")
        return

    # Dynamically create input fields for each feature
    input_data = []

    for feature in feature_labels:
        value = st.text_input(f"{feature}")
        if value.strip():  # Ensure the input is not empty
            try:
                input_data.append(float(value))
            except ValueError:
                st.error(f"Invalid input for {feature}. Please enter a valid number.")
                return

    # When the user presses the button, calculate the probability
    if st.button("Calculate Probability") and len(input_data) == len(feature_labels):
        try:
            # Convert input data to an array and apply the scaler
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # Calculate the probability using the model
            probability = model.predict_proba(input_scaled)[:, 1][0]

            # Display the result
            st.success(f"The probability of fistula is: {probability:.2%}")
        except AttributeError as e:
            st.error(f"Model calculation error: {e}. The model may not be compatible with the current scikit-learn version.")
            st.info("Please regenerate the model or check compatibility.")
        except Exception as e:
            st.error(f"Unexpected error in calculation: {e}")

    # Feature legend
    st.write("\n**Feature Legend:**")
    st.write("- **wLLH_FO_RMAD**: wavelet-LLH Firstorder Robust Mean Absolute Deviation")
    st.write("- **wHLL_GLDM_SDHGLE**: wavelet-HLL Gray Level Dependence Matrix Small Dependence High Gray Level Emphasis")
    st.write("- **o_GLCM_CT**: Original Gray Level Co-occurrence Matrix Cluster Tendency")
    st.write("- **e_GLDM_SDLGLE**: Exponential Gray Level Dependence Matrix Small Dependence Low Gray Level Emphasis")
    st.write("- **e_GLCM_SS**: Exponential Gray Level Co-occurrence Matrix Sum Squares")

if __name__ == "__main__":
    main()

