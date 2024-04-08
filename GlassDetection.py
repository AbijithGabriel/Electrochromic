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

# Function to save the cropped frame as an image
def save_cropped_image(frame_crop, output_path):
    # Specify the directory where you want to save the image
    save_dir = "E:/results"
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Save the cropped frame as an image
    cv2.imwrite(output_path, frame_crop)
    # Enhance the saved image
    enhance_image(output_path)

# Function to enhance the saved image
def enhance_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to find black regions
    _, black_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Apply histogram equalization for contrast enhancement only on black regions
    equalized_black = cv2.equalizeHist(gray, black_mask)

    # Resize equalized_black to match the shape of the black mask
    equalized_black_resized = cv2.resize(equalized_black, (img.shape[1], img.shape[0]))

    # Expand dimensions of equalized_black_resized to match the number of channels in img
    equalized_black_resized = cv2.cvtColor(equalized_black_resized, cv2.COLOR_GRAY2BGR)

    # Replace the black regions in the original image with the enhanced black regions
    img[black_mask == 255] = equalized_black_resized[black_mask == 255]

    # Save the enhanced image
    cv2.imwrite(image_path, img)

# Function to capture frames from the webcam
def capture_from_webcam():
    global video_stream
    
    # Start the video stream
    video_stream = VideoStream(0)
    video_stream.start()

    # Run indefinitely until explicitly stopped
    while True:
        # Check if the video stream has been stopped
        if video_stream.stopped:
            # Break the loop if the stream is stopped
            break
        else:
            # Read a frame from the video stream
            frame = video_stream.read()
            # Convert the frame to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale frame
            rects = detector(gray, 1)
            # Iterate through each detected face
            for i, face_rect in enumerate(rects):
                # Extract coordinates of the detected face
                left = face_rect.left() 
                top = face_rect.top()
                width = face_rect.right() - left
                height = face_rect.bottom() - top

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (left, top), (left+width, top+height), (0,255,0), 2)
                # Label the face with a number
                cv2.putText(frame, f"Face {i+1}", (left - 10, top - 10), font_1, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # Crop the frame to focus on the detected face
                frame_crop = frame[top + 10:top+height-100, left + 30: left+width - 20]

                # Check if the cropped frame has a valid size
                if frame_crop.shape[0] > 0 and frame_crop.shape[1] > 0:
                    # Show the cropped frame
                    cv2.imshow("Cropped Frame", frame_crop)

                    # Smooth the cropped frame
                    img_blur = cv2.GaussianBlur(np.array(frame_crop), (5, 5), sigmaX=1.7, sigmaY=1.7)
                    # Apply Canny edge detection to the cropped frame
                    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
                    # Display the result of Canny edge detection
                    cv2.imshow("Canny Filter", edges)

                    # Extract the center strip of the edges
                    edges_center = edges.T[(int(len(edges.T)/2))]
                    # Check for the presence of white edges indicating glasses
                    if 255 in edges_center:
                        # Display message indicating presence of glasses
                        cv2.rectangle(frame, (left, top+height), (left+width, top+height+40), (0,255,0), cv2.FILLED)
                        cv2.putText(frame, "Glass is Present", (left+10, top+height+20), font_1, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                        # Save the cropped frame as an image when glasses are detected
                        save_cropped_image(frame_crop, f"E:/results/glass_detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    else:
                        # Display message indicating absence of glasses
                        cv2.rectangle(frame, (left, top+height), (left+width, top+height+40), (0,255,0), cv2.FILLED)
                        cv2.putText(frame, "No Glass", (left+10, top+height+20), font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

            # Introduce a delay for processing each frame
            delay = 0.04
            time.sleep(delay)

        # Display the final result frame
        cv2.imshow("Result", frame)
        
        # Check for key press to stop the execution of the program
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Stop capturing video frames
    video_stream.stop()

    # closing all windows 
    cv2.destroyAllWindows()

# Function to capture image from file
def capture_from_file():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    
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
            save_cropped_image(frame_crop, output_path)
            cv2.rectangle(image, (left, top+height), (left+width, top+height+40), (0,255,0), cv2.FILLED)
            cv2.putText(image, "Glass is Present", (left+10, top+height+20), font_1, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (left, top+height), (left+width, top+height+40), (0,255,0), cv2.FILLED)
            cv2.putText(image, "No Glass", (left+10, top+height+20), font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Create the main window
window = tk.Tk()
window.title("Glass Detection")

# Create a frame to hold the buttons
frame = tk.Frame(window)
frame.pack(padx=20, pady=20)

# Create a button to capture frames from webcam
webcam_btn = tk.Button(frame, text="Capture from Webcam", command=capture_from_webcam)
webcam_btn.pack(side=tk.LEFT, padx=10)

# Create a button to capture frames from file
file_btn = tk.Button(frame, text="Capture from File", command=capture_from_file)
file_btn.pack(side=tk.LEFT, padx=10)

# Run the Tkinter event loop
window.mainloop()
