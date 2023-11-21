# gui.py

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import preprocess
import predict
from PIL import Image


def upload_action(event=None):
    filename = filedialog.askopenfilename()
    if filename:
        img = Image.open(filename)
        img.thumbnail((128, 128), Image.Resampling.LANCZOS)  # Resize for display
        photo = ImageTk.PhotoImage(img)
        label_image.config(image=photo)
        label_image.image = photo  # Keep a reference

        # Preprocess and predict
        preprocessed_img = preprocess.preprocess_image(filename)
        digit = predict.predict_digit(preprocessed_img)
        label_prediction.config(text=f'Predicted Digit: {digit}')


# Set up the GUI
root = tk.Tk()
root.title("Digit Recognizer")

# Upload button
button_upload = tk.Button(root, text="Upload Image", command=upload_action)
button_upload.pack()

# Image display
label_image = tk.Label(root)
label_image.pack()

# Prediction display
label_prediction = tk.Label(root, text="Predicted Digit: None", font=("Helvetica", 16))
label_prediction.pack()

root.mainloop()
