import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.layers import TFSMLayer 
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
MODEL_NAME = "brain_tumor_resnet_v1_savedmodel"

SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', MODEL_NAME)

IMG_WIDTH, IMG_HEIGHT = 224, 224
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

try:
    model = TFSMLayer(SAVED_MODEL_PATH, call_endpoint='serve')
    print("Model has been loaded successfully as TFSMLayer.")
    
except Exception as e:
    print(f"CRITICAL ERROR: Model could not be loaded from SavedModel. Checked path: {SAVED_MODEL_PATH}. Error: {e}")
    exit()

preprocess_input = tf.keras.applications.resnet50.preprocess_input

def predict_tumor(pil_image):
    
    if pil_image is None:
        return "LOADING ERROR", "0%", {}
    
    try:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        img_array = image.img_to_array(pil_image.resize((IMG_WIDTH, IMG_HEIGHT)))
        img_array = np.expand_dims(img_array, axis=0)
        
        processed_image = preprocess_input(img_array)
        
        predictions = model(processed_image) 
        
        probs = predictions[0].numpy()
        
        predicted_index = np.argmax(probs)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = probs[predicted_index] * 100
        
        label_dict = {CLASS_NAMES[i]: float(p) * 100 for i, p in enumerate(probs)}
        
        if predicted_class == 'notumor':
            status_message = f"NO TUMOR"
        else:
            status_message = f"TUMOR FOUND: {predicted_class.upper()} TYPE"
            
        return status_message, f"{confidence:.2f}%", label_dict
        
    except Exception as e:
        print(f"PREDICTION ERROR OCCURRED: {e}")
        return "PREDICTION ERROR", "0%", {"Error": str(e)}


interface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil", label="Upload MRI Image", width=300, height=300),
    outputs=[
        gr.Textbox(label="Diagnosis Status"),
        gr.Textbox(label="Confidence Level"),
        gr.JSON(label="Class Probabilities (%)")
    ],
    title="Brain Tumor Classifier",
    description="Classify the MRI images you upload using the trained model.",
    live=False 
)

if __name__ == "__main__":
    interface.launch(share=True)