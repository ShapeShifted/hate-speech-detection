import tkinter as tk
from preprocess import preprocess
from prediction import prediction
import tensorflow as tf
import keras

# Function to display the input string
def show_output():
    input_string = entry.get()
    preprocessed_input_string = preprocess(input_string)
    hate_speech_value = prediction(preprocessed_input_string)
    if (hate_speech_value==1):
        result = 'hate speech'
    else:
        result = 'not hate speech'
    output_label.config(text=f"This message is {result}.")

# Create the main window
root = tk.Tk()
root.title("Message Detection System")

# Create and place the input label
label = tk.Label(root, text="Submit any message:")
label.pack(pady=10)

# Create and place the entry widget
entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# Create and place the submit button
button = tk.Button(root, text="Submit", command=show_output)
button.pack(pady=10)

# Create and place the output label
output_label = tk.Label(root, text="")
output_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
