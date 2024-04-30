import dlib
import numpy as np
import cv2
import os
import time
import tkinter as tk
from tkinter import filedialog
from threading import Thread

# Define the font for text rendering
font_1 = cv2.FONT_HERSHEY_SIMPLEX

# Declare the dlib frontal face detector
detector = dlib.get_frontal_face_detector()

class VideoStream:
    def __init__(self, stream):
        self.video = cv2.VideoCapture(stream)
        # Setting the FPS for the video stream
        self.video.set(cv2.CAP_PROP_FPS, 60)

        if not self.video.isOpened():
            print("Can't access the webcam stream.")
            exit(0)
        
        # Read the first frame from the video stream
        self.grabbed, self.frame = self.video.read()
        self.stopped = True
        # Create a thread for updating the video stream
        self.thread = Thread(target=self.update)
        # Set the thread as a daemon thread
        self.thread.daemon = True
    
    def start(self):
        self.stopped = False
        # Start the thread for updating the video stream
        self.thread.start()

    def update(self):
        while True:
            if self.stopped:
                break
            # Read the next frame from the video stream
            self.grabbed, self.frame = self.video.read()
        self.video.release()

    def read(self):
        # Return the current frame from the video stream
        return self.frame

    def stop(self):
        # Set the stopped flag to True, indicating that the video stream should stop
        self.stopped = True

# Function to save the detected face as an image
def save_cropped_image(frame_crop, output_path):
    # Specify the directory where you want to save the image
    save_dir = "E:/results"
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Save the detected face as an image
    cv2.imwrite(output_path, frame_crop)

# Function to capture frames from the webcam
def capture_from_webcam():
    cap = cv2.VideoCapture(0)  # Open the external webcam (change the index if needed)

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam

        if not ret:
            print("Error: Unable to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)
        for i, face_rect in enumerate(rects):
            left = face_rect.left() 
            top = face_rect.top()
            width = face_rect.right() - left
            height = face_rect.bottom() - top

            cv2.rectangle(frame, (left, top), (left+width, top+height), (0,255,0), 2)
            cv2.putText(frame, f"Face {i+1}", (left - 10, top - 10), font_1, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            frame_crop = frame[top + 10:top+height-100, left + 30: left+width - 20]

            if frame_crop is not None and frame_crop.size != 0:
                img_blur = cv2.GaussianBlur(np.array(frame_crop), (5, 5), sigmaX=1.7, sigmaY=1.7)
                edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

                edges_center = edges.T[(int(len(edges.T)/2))]
                if 255 in edges_center:
                    output_path = f"E:/results/glass_detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.rectangle(frame, (left, top+height), (left+width, top+height+40), (0,255,0), cv2.FILLED)
                    cv2.putText(frame, "Glass is Present", (left+10, top+height+20), font_1, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    frame_crop_output_path = f"E:/results/frame_crop_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(frame_crop_output_path, frame_crop)  # Save the frame crop
                else:
                    cv2.rectangle(frame, (left, top+height), (left+width, top+height+40), (0,255,0), cv2.FILLED)
                    cv2.putText(frame, "No Glass", (left+10, top+height+20), font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to capture image from file
def capture_from_file():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", ".jpg"), ("all files", ".*")))
    
    if not file_path:
        # If no file is selected, return
        return

    image = cv2.imread(file_path)

    if image is None or image.size == 0:
        print("Error: Unable to open or read the file.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    for i, face_rect in enumerate(rects):
        left = face_rect.left() 
        top = face_rect.top()
        width = face_rect.right() - left
        height = face_rect.bottom() - top

        cv2.rectangle(image, (left, top), (left+width, top+height), (0,255,0), 2)
        cv2.putText(image, f"Face {i+1}", (left - 10, top - 10), font_1, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        frame_crop = image[top + 10:top+height-100, left + 30: left+width - 20]

        img_blur = cv2.GaussianBlur(np.array(frame_crop), (5, 5), sigmaX=1.7, sigmaY=1.7)
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

        edges_center = edges.T[(int(len(edges.T)/2))]
        if 255 in edges_center:
            output_path = f"E:/results/glass_detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            # save_cropped_image(frame_crop, output_path)
            cv2.rectangle(image, (left, top+height), (left+width, top+height+40), (0,255,0), cv2.FILLED)
            cv2.putText(image, "Glass is Present", (left+10, top+height+20), font_1, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            # Extract the full face region if glass is present
            frame_crop_output_path = f"E:/results/frame_crop_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(frame_crop_output_path, frame_crop)  # Save the frame crop
        else:
            cv2.rectangle(image, (left, top+height), (left+width, top+height+40), (0,255,0), cv2.FILLED)
            cv2.putText(image, "No Glass", (left+10, top+height+20), font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def reduce_sunglasses_tint_dcp(frame):
    # Convert BGR image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Estimate the dark channel of the image
    dark_channel = cv2.erode(gray, np.ones((600,600), np.uint8))
    
    # Apply dark channel prior to adjust colors
    adjusted_frame = frame - dark_channel[..., np.newaxis]
    
    # Enhance contrast to improve visibility of eyes
    adjusted_frame = cv2.convertScaleAbs(adjusted_frame, alpha=2.1, beta=9)

    return adjusted_frame

def capture_from_file_and_reduce_tint():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    
    if not file_path:
        # If no file is selected, return
        return

    frame = cv2.imread(file_path)

    if frame is None or frame.size == 0:
        print("Error: Unable to open or read the file.")
        return

    # Apply the tint reduction function to the image
    adjusted_frame = reduce_sunglasses_tint_dcp(frame)
    
    # Display the adjusted frame
    cv2.imshow("Tint Reduced Frame", adjusted_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2
from tkinter import filedialog

# Load the pre-trained Haar Cascade classifier for sunglasses detection
sunglasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# Function to reduce the tint of sunglasses using the Haar filter method
def reduce_sunglasses_tint_haar(frame):
    # Convert the frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect sunglasses in the grayscale frame
    sunglasses = sunglasses_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # If sunglasses are detected, reduce the tint
    if len(sunglasses) > 0:
        for (x, y, w, h) in sunglasses:
            # Extract the region of interest (sunglasses) from the frame
            sunglasses_roi = frame[y:y+h, x:x+w]
            
            # Apply a Haar filter method to reduce tint (example)
            # You can replace this with your specific Haar filter method
            # For example, you can apply a custom image processing technique
            
            # Example: apply Gaussian blur to the sunglasses region
            sunglasses_roi = cv2.GaussianBlur(sunglasses_roi, (25, 25), 0)
            
            # Replace the sunglasses region in the frame with the tint-reduced region
            frame[y:y+h, x:x+w] = sunglasses_roi
    
    return frame

# Function to capture image from file and apply Haar filter for reducing tint
def capture_from_file_and_reduce_tint_haar():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    
    if not file_path:
        # If no file is selected, return
        return

    frame = cv2.imread(file_path)

    if frame is None or frame.size == 0:
        print("Error: Unable to open or read the file.")
        return

    # Apply the Haar filter method to reduce tint
    frame_with_tint_reduced = reduce_sunglasses_tint_haar(frame)
    
    # Display the adjusted frame
    cv2.imshow("Frame with Tint Reduced (Haar Filter)", frame_with_tint_reduced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2
import numpy as np
from tkinter import filedialog

# Define the function to reduce tint using histogram equalization
def reduce_sunglasses_tint_with_hist_equalization(image):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply histogram equalization to the L channel
    l_channel_eq = cv2.equalizeHist(l_channel)
    
    # Merge the equalized L channel with the original A and B channels
    lab_image_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
    
    # Convert LAB image back to BGR color space
    final_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2BGR)
    
    # Increase contrast
    final_image = cv2.convertScaleAbs(final_image, alpha=1.1, beta=3)
    
    return final_image

# Function to capture image from file and apply histogram equalization for reducing tint
def capture_from_file_and_reduce_tint_with_hist_equalization():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    
    if not file_path:
        # If no file is selected, return
        return

    image = cv2.imread(file_path)

    if image is None or image.size == 0:
        print("Error: Unable to open or read the file.")
        return

    # Apply the histogram equalization method to reduce tint
    equalized_image = reduce_sunglasses_tint_with_hist_equalization(image)
    
    # Display the adjusted image
    cv2.imshow("Image with Tint Reduced (Histogram Equalization)", equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



import cv2
import numpy as np
from tkinter import filedialog

# Define the function to reduce tint using histogram equalization
def reduce_sunglasses_tint_hsv_conversion(image):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    L, A, B = cv2.split(lab)
    
    # Apply histogram equalization to the L channel
    L_eq = cv2.equalizeHist(L)
    
    # Merge the equalized L channel with the original A and B channels
    lab_eq = cv2.merge((L_eq, A, B))
    
    # Convert LAB image back to BGR color space
    adjusted_image = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    return adjusted_image

# Function to capture image from file and apply histogram equalization for reducing tint
def capture_from_file_and_reduce_tint_hsv_conversion():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    
    if not file_path:
        # If no file is selected, return
        return

    image = cv2.imread(file_path)

    if image is None or image.size == 0:
        print("Error: Unable to open or read the file.")
        return

    # Apply the histogram equalization method to reduce tint
    adjusted_image = reduce_sunglasses_tint_hsv_conversion(image)
    
    # Display the adjusted image
    cv2.imshow("Image with Tint Reduced (Histogram Equalization)", adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2
import numpy as np
from tkinter import filedialog

# Define the function to reduce tint using thresholding, masking, and blending
def reduce_sunglasses_tint_threshold(image):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    L, A, B = cv2.split(lab_image)
    
    # Thresholding to create a mask for the sunglasses (assuming they are dark)
    mask = cv2.threshold(B, 30, 255, cv2.THRESH_BINARY)[1]
    
    # Invert the mask to select non-sunglasses regions
    mask_inv = cv2.bitwise_not(mask)
    
    # Increase brightness of the B channel in the sunglasses region
    adjusted_B = cv2.add(B, 12, mask=mask)
    
    # Merge adjusted B channel with original LAB channels
    lab_adjusted = cv2.merge((L, A, adjusted_B))
    
    # Convert LAB adjusted image back to BGR color space
    adjusted_image = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
    
    return adjusted_image

# Function to capture image from file and apply thresholding, masking, and blending for tint reduction
def capture_from_file_and_reduce_tint_threshold_mask():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    
    if not file_path:
        # If no file is selected, return
        return

    image = cv2.imread(file_path)

    if image is None or image.size == 0:
        print("Error: Unable to open or read the file.")
        return

    # Apply the thresholding, masking, and blending method to reduce tint
    adjusted_image = reduce_sunglasses_tint_threshold(image)
    
    # Display the adjusted image
    cv2.imshow("Image with Tint Reduced (Thresholding, Masking, and Blending)", adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2
from tkinter import filedialog

# Define the function to reduce tint using histogram equalization
def reduce_sunglasses_tint_with_hist_equalization_2(image):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply histogram equalization to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel_eq = clahe.apply(l_channel)
    
    # Merge the equalized L channel with the original A and B channels
    lab_image_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
    
    # Convert LAB image back to BGR color space
    final_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2BGR)
    
    return final_image

# Function to capture image from file and apply histogram equalization for tint reduction
def capture_from_file_and_reduce_tint_with_hist_equalization_2():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    
    if not file_path:
        # If no file is selected, return
        return

    image = cv2.imread(file_path)

    if image is None or image.size == 0:
        print("Error: Unable to open or read the file.")
        return

    # Apply the histogram equalization method to reduce tint
    adjusted_image = reduce_sunglasses_tint_with_hist_equalization_2(image)
    
    # Display the adjusted image
    cv2.imshow("Image with Tint Reduced (Contrast Equalization - CLAHE)", adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





# Create the main window
from customtkinter import *
from PIL import Image

def show_home_content():
    # Clear previous content and display home content
    clear_main_view()
    CTkButton(master=main_view, text="Capture from Webcam", font=("Arial", 20), width=250, height=60, command=capture_from_webcam).pack(pady=90)
    CTkButton(master=main_view, text="Capture from File", font=("Arial", 20), width=250, height=60, command=capture_from_file).pack(pady=40)

# Rest of the code remains the same


def show_about_content():
    # Clear previous content and display about content
    clear_main_view()
    CTkLabel(master=main_view, text="About Project", font=("Arial", 20)).pack()


def show_algorithms_content():
    # Clear previous content and display algorithms content
    clear_main_view()

    # Create a frame for the first row of buttons
    first_row_frame = CTkFrame(master=main_view, fg_color="#fff")
    first_row_frame.pack()

    # Create buttons for the first row
    CTkButton(master=first_row_frame, text="Dark Channel Prior", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint).pack(side="left", padx=10, pady=50)
    
    CTkButton(master=first_row_frame, text="haar filter", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_haar).pack(side="left", padx=10, pady=50)

    # Create a frame for the second row of buttons
    second_row_frame = CTkFrame(master=main_view, fg_color="#fff")
    second_row_frame.pack()

    # Create buttons for the second row
    CTkButton(master=second_row_frame, text="histogram equalization", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_with_hist_equalization).pack(side="left", padx=10, pady=50)
    CTkButton(master=second_row_frame, text="HSV conversion technique", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_hsv_conversion).pack(side="left", padx=10, pady=10)

    Third_row_frame = CTkFrame(master=main_view, fg_color="#fff")
    Third_row_frame.pack()
    CTkButton(master=Third_row_frame, text="thresholding, masking, and blending", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_threshold_mask).pack(side="left", padx=10, pady=50)
    CTkButton(master=Third_row_frame, text="Contrast Equalization (CLAHE)", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_with_hist_equalization_2).pack(side="left", padx=10, pady=10)


def show_returns_content():
    # Clear previous content and display returns content
    clear_main_view()
    CTkLabel(master=main_view, text="Returns", font=("Arial", 20)).pack()

def clear_main_view():
    # Clear all widgets in the main_view frame
    for widget in main_view.winfo_children():
        widget.destroy()

app = CTk()
app.geometry("856x645")
app.resizable(0,0)

set_appearance_mode("light")

sidebar_frame = CTkFrame(master=app, fg_color="#2A8C55",  width=176, height=650, corner_radius=0)
sidebar_frame.pack_propagate(0)
sidebar_frame.pack(fill="y", anchor="w", side="left")

logo_img_data = Image.open("sunglasses.png")
logo_img = CTkImage(dark_image=logo_img_data, light_image=logo_img_data, size=(77.68, 85.42))

CTkLabel(master=sidebar_frame, text="", image=logo_img).pack(pady=(38, 0), anchor="center")

# Function to show home content when home button is clicked
CTkButton(master=sidebar_frame, text="Home", fg_color="transparent", font=("Arial Bold", 14), hover_color="#207244", anchor="w", command=show_home_content).pack(anchor="center", ipady=5, pady=(60, 0))

# Function to show about content when about button is clicked
CTkButton(master=sidebar_frame, text="Tint Reduce", fg_color="transparent", font=("Arial Bold", 14), hover_color="#207244", anchor="w", command=show_algorithms_content).pack(anchor="center", ipady=5, pady=(16, 0))

# Function to show team content when team button is clicked
CTkButton(master=sidebar_frame, text="About", fg_color="transparent", font=("Arial Bold", 14), hover_color="#207244", anchor="w", command=show_about_content).pack(anchor="center", ipady=5, pady=(16, 0))


main_view = CTkFrame(master=app, fg_color="#fff",  width=680, height=650, corner_radius=0)
main_view.pack_propagate(0)
main_view.pack(side="left")

show_home_content()  # Show home content by default

app.mainloop()
