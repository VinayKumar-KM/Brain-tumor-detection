import streamlit as st
from session_state import get_session_state
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Function to load the trained model
def load_brain_tumor_model(model_path):
    try:
        model = load_model(model_path)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e
    return model

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize image to match model's expected sizing
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to predict tumor or non-tumor
def predict_tumor(model, img_array):
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input
    try:
        prediction = model.predict(img_array)
        st.write("Prediction successful.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        raise e
    return prediction  # Returns a 2D array with probabilities

# Function to display login page
def login_page():
    session_state = get_session_state()
    if session_state.logged_in:
        return True
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "12345":
            session_state.logged_in = True
            st.success("Logged In as Admin")
            return True
        else:
            st.error("Incorrect Username or Password")
    return False

# Function to plot prediction results as a bar chart
def plot_prediction_bar(tumor_probability):
    labels = ['Tumor', 'No Tumor']
    probabilities = [tumor_probability, 100 - tumor_probability]
    colors = ['red', 'green']

    fig, ax = plt.subplots()
    ax.bar(labels, probabilities, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Probability (%)')
    ax.set_title('Tumor Detection Probability')

    for i, v in enumerate(probabilities):
        ax.text(i, v + 1, f"{v:.2f}%", color='black', ha='center')

    return fig

# Main function to run the Streamlit app
def main():
    st.title("Brain Tumor Detection App")
    st.image('img1.jpeg', width=750)
    model_path = 'model_brain_tumor.h5'  # Replace with your actual model path
    model = load_brain_tumor_model(model_path)
    
    # Display login page
    if login_page():
        st.subheader("Upload MRI Image for Tumor Detection")
        uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded MRI.', use_column_width=True)
            
            # Preprocess the image
            img_array = preprocess_image(image)
            st.write("")
            st.write("Classifying...")
            
            # Make prediction
            try:
                prediction = predict_tumor(model, img_array)
                tumor_probability = prediction[0][0] * 100  # Convert to percentage
                
                # Display prediction result
                if tumor_probability >= 50:
                    st.warning(f"Prediction: Tumor detected with probability {tumor_probability:.2f}%")
                else:
                    st.success(f"Prediction: No tumor detected with probability {100 - tumor_probability:.2f}%")
                
                # Display the bar chart
                st.subheader("Bar Chart")
                bar_fig = plot_prediction_bar(tumor_probability)
                st.pyplot(bar_fig)
                    
            except Exception as e:
                st.error(f"Error in prediction: {e}")

# Run the main function
if __name__ == "__main__":
    main()
