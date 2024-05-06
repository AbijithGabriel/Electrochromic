import dlib
import numpy as np
import cv2
import os
import time
import tkinter as tk
from tkinter import filedialog
from threading import Thread
from tkinter import filedialog
from customtkinter import *
from PIL import Image
import math
from PIL import Image, ImageTk

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


import math

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

def calculate_psnr(original_image, adjusted_image):
    # Convert images to float32
    original_image = original_image.astype(np.float32)
    adjusted_image = adjusted_image.astype(np.float32)

    # Compute MSE
    mse = np.mean((original_image - adjusted_image) ** 2)

    if mse == 0:
        return float('inf')

    # Maximum possible pixel value
    max_pixel_value = 255.0

    # Calculate PSNR
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)

    return psnr

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

    # Resize the adjusted frame
    scale_percent = 200  # percent of original size
    width = int(adjusted_frame.shape[1] * scale_percent / 100)
    height = int(adjusted_frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    adjusted_frame_resized = cv2.resize(adjusted_frame, dim, interpolation=cv2.INTER_AREA)

    # Calculate PSNR
    original_frame = cv2.imread(file_path)
    psnr = calculate_psnr(original_frame, adjusted_frame)
    psnr_text = f"PSNR: {psnr:.2f}"

    # Overlay PSNR value on the image
    cv2.putText(adjusted_frame_resized, psnr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the adjusted frame
    cv2.imshow("Original image", original_frame)
    cv2.imshow("Tint Reduced Frame", adjusted_frame_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def calculate_psnr(original_image, adjusted_image):
    # Convert images to float32
    original_image = original_image.astype(np.float32)
    adjusted_image = adjusted_image.astype(np.float32)

    # Compute MSE
    mse = np.mean((original_image - adjusted_image) ** 2)

    if mse == 0:
        return float('inf')

    # Maximum possible pixel value
    max_pixel_value = 255.0

    # Calculate PSNR
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)

    return psnr

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
    
    # Calculate PSNR
    original_frame = cv2.imread(file_path)
    psnr = calculate_psnr(original_frame, frame_with_tint_reduced)
    psnr_text = f"PSNR: {psnr:.2f}"

    # Overlay PSNR value on the image
    cv2.putText(frame_with_tint_reduced, psnr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the adjusted frame
    cv2.imshow("orginal image ", original_frame)
    cv2.imshow("Frame with Tint Reduced (Haar Filter)", frame_with_tint_reduced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# Define the function to reduce tint using histogram equalization
import math

def calculate_psnr(original_image, adjusted_image):
    # Convert images to float32
    original_image = original_image.astype(np.float32)
    adjusted_image = adjusted_image.astype(np.float32)

    # Compute MSE
    mse = np.mean((original_image - adjusted_image) ** 2)

    if mse == 0:
        return float('inf')

    # Maximum possible pixel value
    max_pixel_value = 255.0

    # Calculate PSNR
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)

    return psnr

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
    
    # Calculate PSNR
    original_image = cv2.imread(file_path)
    psnr = calculate_psnr(original_image, equalized_image)
    psnr_text = f"PSNR: {psnr:.2f}"

    # Overlay PSNR value on the image
    cv2.putText(equalized_image, psnr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the adjusted image
    cv2.imshow("Original image", original_image)
    cv2.imshow("Image with Tint Reduced (Histogram Equalization)", equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def calculate_psnr(original_image, adjusted_image):
    # Convert images to float32
    original_image = original_image.astype(np.float32)
    adjusted_image = adjusted_image.astype(np.float32)

    # Compute MSE
    mse = np.mean((original_image - adjusted_image) ** 2)

    if mse == 0:
        return float('inf')

    # Maximum possible pixel value
    max_pixel_value = 255.0

    # Calculate PSNR
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)

    return psnr

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
    
    # Calculate PSNR
    original_image = cv2.imread(file_path)
    psnr = calculate_psnr(original_image, adjusted_image)
    psnr_text = f"PSNR: {psnr:.2f}"

    # Overlay PSNR value on the image
    cv2.putText(adjusted_image, psnr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the adjusted image
    cv2.imshow("Original image", original_image)
    cv2.imshow("Image with Tint Reduced (hsv conversion)", adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# Define the function to reduce tint using thresholding, masking, and blending


def calculate_psnr(original_image, adjusted_image):
    # Convert images to float32
    original_image = original_image.astype(np.float32)
    adjusted_image = adjusted_image.astype(np.float32)

    # Compute MSE
    mse = np.mean((original_image - adjusted_image) ** 2)

    if mse == 0:
        return float('inf')

    # Maximum possible pixel value
    max_pixel_value = 255.0

    # Calculate PSNR
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)

    return psnr

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
    
    # Calculate PSNR
    original_image = cv2.imread(file_path)
    psnr = calculate_psnr(original_image, adjusted_image)
    psnr_text = f"PSNR: {psnr:.2f}"

    # Overlay PSNR value on the image
    cv2.putText(adjusted_image, psnr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the adjusted image
    cv2.imshow("Original image", original_image)
    cv2.imshow("Image with Tint Reduced (Thresholding, Masking, and Blending)", adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# Define the function to reduce tint using histogram equalization


def calculate_psnr(original_image, adjusted_image):
    # Convert images to float32
    original_image = original_image.astype(np.float32)
    adjusted_image = adjusted_image.astype(np.float32)

    # Compute MSE
    mse = np.mean((original_image - adjusted_image) ** 2)

    if mse == 0:
        return float('inf')

    # Maximum possible pixel value
    max_pixel_value = 255.0

    # Calculate PSNR
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)

    return psnr

def reduce_sunglasses_tint_with_hist_equalization_2(image):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply histogram equalization to the L channel using CLAHE
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
    
    # Calculate PSNR
    original_image = cv2.imread(file_path)
    psnr = calculate_psnr(original_image, adjusted_image)
    psnr_text = f"PSNR: {psnr:.2f}"

    # Overlay PSNR value on the image
    cv2.putText(adjusted_image, psnr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the adjusted image
    cv2.imshow("Original image", original_image)
    cv2.imshow("Image with Tint Reduced (Contrast Equalization - CLAHE)", adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def calculate_psnr(original_image, adjusted_image):
    # Convert images to float32
    original_image = original_image.astype(np.float32)
    adjusted_image = adjusted_image.astype(np.float32)

    # Compute MSE
    mse = np.mean((original_image - adjusted_image) ** 2)

    if mse == 0:
        return float('inf')

    # Maximum possible pixel value
    max_pixel_value = 255.0

    # Calculate PSNR
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)

    return psnr

def reduce_sunglasses_tint_with_gaussian_filter(image):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    L, A, B = cv2.split(lab_image)
    
    # Apply Gaussian filter to the B channel to reduce tint
    B_filtered = cv2.GaussianBlur(B, (25, 25), 0)
    
    # Merge the filtered B channel with the original LAB channels
    lab_adjusted = cv2.merge((L, A, B_filtered))
    
    # Convert LAB adjusted image back to BGR color space
    adjusted_image = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
    
    return adjusted_image

# Function to capture image from file and apply Gaussian filter for reducing tint
def capture_from_file_and_reduce_tint_with_gaussian_filter():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    
    if not file_path:
        # If no file is selected, return
        return

    image = cv2.imread(file_path)

    if image is None or image.size == 0:
        print("Error: Unable to open or read the file.")
        return

    # Apply the Gaussian filter method to reduce tint
    adjusted_image = reduce_sunglasses_tint_with_gaussian_filter(image)
    
    # Calculate PSNR
    original_image = cv2.imread(file_path)
    psnr = calculate_psnr(original_image, adjusted_image)
    psnr_text = f"PSNR: {psnr:.2f}"

    # Overlay PSNR value on the image
    cv2.putText(adjusted_image, psnr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the adjusted image
    cv2.imshow("Original image", original_image)
    cv2.imshow("Image with Tint Reduced (Gaussian Filter)", adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#8

def calculate_psnr(original_image, adjusted_image):
    # Convert images to float32
    original_image = original_image.astype(np.float32)
    adjusted_image = adjusted_image.astype(np.float32)

    # Compute MSE
    mse = np.mean((original_image - adjusted_image) ** 2)

    if mse == 0:
        return float('inf')

    # Maximum possible pixel value
    max_pixel_value = 255.0

    # Calculate PSNR
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)

    return psnr

def reduce_sunglasses_tint_with_gray_world(image):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Compute average values for each channel
    avg_l = np.mean(l_channel)
    avg_a = np.mean(a_channel)
    avg_b = np.mean(b_channel)
    
    # Compute scaling factors
    max_avg = max(avg_l, avg_a, avg_b)
    scale_l = max_avg / avg_l
    scale_a = max_avg / avg_a
    scale_b = max_avg / avg_b
    
    # Scale LAB channels
    l_channel_scaled = np.clip(l_channel * scale_l, 0, 255).astype(np.uint8)
    a_channel_scaled = np.clip(a_channel * scale_a, 0, 255).astype(np.uint8)
    b_channel_scaled = np.clip(b_channel * scale_b, 0, 255).astype(np.uint8)
    
    # Merge the scaled LAB channels
    lab_adjusted = cv2.merge((l_channel_scaled, a_channel_scaled, b_channel_scaled))
    
    # Convert LAB adjusted image back to BGR color space
    adjusted_image = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
    
    return adjusted_image

def capture_from_file_and_reduce_tint_with_gray_world():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    
    if not file_path:
        # If no file is selected, return
        return

    image = cv2.imread(file_path)

    if image is None or image.size == 0:
        print("Error: Unable to open or read the file.")
        return

    # Apply the Gray World algorithm to reduce tint
    adjusted_image = reduce_sunglasses_tint_with_gray_world(image)
    
    # Calculate PSNR
    original_image = cv2.imread(file_path)
    psnr = calculate_psnr(original_image, adjusted_image)
    print("PSNR:", psnr)

    # Display the adjusted image
    cv2.imshow("Original image", original_image)
    cv2.imshow("Image with Tint Reduced (Gray World)", adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Create the main window

from PIL import Image, ImageTk

def show_home_content():
    # Clear previous content and display home content
    clear_main_view()

    # Load the image for the buttons
    webcam_image_path = "webcam.png"  # Replace with the actual path to your webcam image
    file_image_path = "folder.png"  # Replace with the actual path to your file image

    # Load and resize the images
    webcam_image = Image.open(webcam_image_path).resize((40, 40))
    file_image = Image.open(file_image_path).resize((40, 40))

    # Convert images for Tkinter
    webcam_photo = ImageTk.PhotoImage(webcam_image)
    file_photo = ImageTk.PhotoImage(file_image)

    # Create the button for capturing from webcam
    webcam_button = CTkButton(
        master=main_view,
        text="Capture from Webcam",
        font=("Arial", 20),
        width=250,
        height=60,
        command=capture_from_webcam,
        image=webcam_photo,  # Set the image for the button
        compound="left"  # Place the image to the left of the text
    )
    webcam_button.image = webcam_photo  # Keep a reference to avoid garbage collection
    webcam_button.pack(pady=90)

    # Create the button for capturing from file
    file_button = CTkButton(
        master=main_view,
        text="Capture from File     ",
        font=("Arial", 20),
        width=250,
        height=60,
        command=capture_from_file,
        image=file_photo,  # Set the image for the button
        compound="left"  # Place the image to the left of the text
    )
    file_button.image = file_photo  # Keep a reference to avoid garbage collection
    file_button.pack(pady=40)




def show_about_content():
    # Clear previous content and display about content
    clear_main_view()
    CTkLabel(master=main_view, text="About Project", font=("Arial", 20)).pack()
    
    metrics_frame = CTkFrame(master=main_view, fg_color="transparent")
    metrics_frame.pack(anchor="n", fill="x",  padx=27, pady=(36, 0))

    shipped_metric = CTkFrame(master=metrics_frame, fg_color="#2A8C55", width=200, height=60)
    shipped_metric.grid_propagate(0)
    shipped_metric.pack(side="left",expand=True, anchor="center")

    shipping_img_data = Image.open("webcam.png")
    shipping_img = CTkImage(light_image=shipping_img_data, dark_image=shipping_img_data, size=(43, 43))

    CTkLabel(master=shipped_metric, image=shipping_img, text="").grid(row=0, column=0, rowspan=2, padx=(12,5), pady=10)

    CTkLabel(master=shipped_metric, text="Shipping", text_color="#fff", font=("Arial Black", 15)).grid(row=0, column=1, sticky="sw")
    CTkLabel(master=shipped_metric, text="91", text_color="#fff",font=("Arial Black", 15), justify="left").grid(row=1, column=1, sticky="nw", pady=(0,10))
 


def show_algorithms_content():
    # Clear previous content and display algorithms content
    clear_main_view()

    # Create a frame for the first row of buttons
    first_row_frame = CTkFrame(master=main_view, fg_color="#fff")
    first_row_frame.pack()

    # Create buttons for the first row
    CTkButton(master=first_row_frame, text="Dark Channel Prior", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint).pack(side="left", padx=10, pady=40)
    CTkButton(master=first_row_frame, text="haar filter", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_haar).pack(side="left", padx=10, pady=40)

    # Create a frame for the second row of buttons
    second_row_frame = CTkFrame(master=main_view, fg_color="#fff")
    second_row_frame.pack()

    # Create buttons for the second row
    CTkButton(master=second_row_frame, text="histogram equalization", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_with_hist_equalization).pack(side="left", padx=10, pady=40)
    CTkButton(master=second_row_frame, text="HSV conversion technique", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_hsv_conversion).pack(side="left", padx=10, pady=40)

    Third_row_frame = CTkFrame(master=main_view, fg_color="#fff")
    Third_row_frame.pack()
    CTkButton(master=Third_row_frame, text="thresholding, masking, and blending", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_threshold_mask).pack(side="left", padx=10, pady=40)
    CTkButton(master=Third_row_frame, text="Contrast Equalization (CLAHE)", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_with_hist_equalization_2).pack(side="left", padx=10, pady=40)
    
    fourth_row_frame = CTkFrame(master=main_view, fg_color="#fff")
    fourth_row_frame.pack()
    CTkButton(master=fourth_row_frame, text="Gaussian filter", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_with_gaussian_filter).pack(side="left", padx=10, pady=40)
    CTkButton(master=fourth_row_frame, text=" Color constancy", font=("Arial", 20), width=250, height=60,command=capture_from_file_and_reduce_tint_with_gray_world).pack(side="left", padx=10, pady=40)


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
CTkButton(master=sidebar_frame, text="Home", fg_color="transparent", font=("Arial Bold", 18), hover_color="#207244", anchor="w", command=show_home_content).pack(anchor="center", ipady=5, pady=(60, 0))

# Function to show about content when about button is clicked
CTkButton(master=sidebar_frame, text="Tint Reduce", fg_color="transparent", font=("Arial Bold", 18), hover_color="#207244", anchor="w", command=show_algorithms_content).pack(anchor="center", ipady=5, pady=(16, 0))

# Function to show team content when team button is clicked
CTkButton(master=sidebar_frame, text="About", fg_color="transparent", font=("Arial Bold", 18), hover_color="#207244", anchor="w", command=show_about_content).pack(anchor="center", ipady=5, pady=(16, 0))


main_view = CTkFrame(master=app, fg_color="#fff",  width=680, height=650, corner_radius=0)
main_view.pack_propagate(0)
main_view.pack(side="left")




show_home_content()  # Show home content by default

app.mainloop()
