import tensorflow as tf
import cv2

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

root = tk.Tk()
root.geometry("300x400")
model = tf.keras.models.load_model("tumor_model.keras")
canvas = tk.Canvas(root, width=300, height=300)
text = tk.Label(root, text="")
label = tk.Label(root, text="")


def choose_file():
    global text, label
    file_path = filedialog.askopenfilename()
    text.pack_forget()
    label.pack_forget()
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(file_path, gray)
    new_image = cv2.imread(file_path)
    new_image = cv2.resize(image, (30, 30))
    cv2.imwrite(file_path, image)
    i = Image.open(file_path)
    i = i.resize((300, 300))
    photo = ImageTk.PhotoImage(i)

    label = tk.Label(image=photo)
    label.image = photo  
    label.pack()
    classification = model.predict([new_image.reshape(1, 30, 30, 3)]).argmax()
    if classification == 1:
        text = tk.Label(root, text="No tumor detected", font=("Arial", 30))
    if classification == 0:
        text = tk.Label(root, text="Tumor detected", font=("Arial", 30))

    text.pack()


b = tk.Button(root, text="Choose File", command=choose_file, width=20, height=1)
b.pack()

root.mainloop()


