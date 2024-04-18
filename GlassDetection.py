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
def save_face_image(face_image, output_path):
    # Specify the directory where you want to save the image
    save_dir = "E:/results"
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Save the detected face as an image
    cv2.imwrite(output_path, face_image)
    # Enhance the saved image
    enhance_image(output_path)

# Function to enhance the saved image
def enhance_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Increase brightness and contrast
    alpha = 1.9  # Contrast control (1.0-3.0)
    beta = 8  # Brightness control (0-100)
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Save the enhanced image
    cv2.imwrite(image_path, enhanced)

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
                    face_image = frame[top:top+height, left:left+width]
                    if face_image is not None and face_image.size != 0:
                        save_face_image(face_image, output_path)  # Save the face image
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
            face_image = image[top:top+height, left:left+width]
            if face_image is not None and face_image.size != 0:
                save_face_image(face_image, output_path)  # Save the face image
        else:
            cv2.rectangle(image, (left, top+height), (left+width, top+height+40), (0,255,0), cv2.FILLED)
            cv2.putText(image, "No Glass", (left+10, top+height+20), font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Create the main window
window = tk.Tk()
window.title("ELECTROCHROMIC GLASSES")
window.configure(bg="#525354")  # Set background color to light gray

# Create a frame to hold the buttons
frame = tk.Frame(window, bg="#f0f0f0")  # Set background color of the frame to match the window
frame.pack(padx=300, pady=300)


# Create a button to capture frames from webcam
webcam_btn = tk.Button(frame, 
                       text="Capture from Webcam", 
                       command=capture_from_webcam, 
                       width=20, 
                       height=3, 
                       font=("Helvetica", 15),  # Specify the font family and size
                       bg="#161716",            # Background color (blue)
                       fg="#ffffff",            # Text color (white)
                       bd=0,                    # Border width (0 for no border)
                       cursor="hand2",          # Cursor style (hand pointer)
                       activebackground="#0056b3",  # Background color when button is pressed
                       activeforeground="#ffffff",  # Text color when button is pressed
                       padx=10,                 # Horizontal padding
                       pady=5)                  # Vertical padding
webcam_btn.pack(side=tk.LEFT, padx=10, pady=10)


# Create a button to capture frames from file
file_btn = tk.Button(  frame,
                       text="Capture from File", 
                       command=capture_from_file,
                       width=20, 
                       height=3, 
                       font=("Helvetica", 15),  # Specify the font family and size
                       bg="#161716",            # Background color (blue)
                       fg="#ffffff",            # Text color (white)
                       bd=0,                    # Border width (0 for no border)
                       cursor="hand2",          # Cursor style (hand pointer)
                       activebackground="#0056b3",  # Background color when button is pressed
                       activeforeground="#ffffff",  # Text color when button is pressed
                       padx=10,                 # Horizontal padding
                       pady=5)
file_btn.pack(side=tk.LEFT, padx=10)

# Run the Tkinter event loop
window.mainloop()
