# import dlib
# import numpy as np
# from threading import Thread
# import cv2
# import time

# # a cv2 font type
# font_1 = cv2.FONT_HERSHEY_SIMPLEX
# # Declare dlib frontal face detector
# detector = dlib.get_frontal_face_detector()

# # This class handles the video stream 
# # captured from the WebCam
# class VideoStream:
#     def __init__(self, stream):
#         self.video = cv2.VideoCapture(stream)
#         # Setting the FPS for the video stream
#         self.video.set(cv2.CAP_PROP_FPS, 60)

#         if self.video.isOpened() is False:
#             print("Can't accessing the webcam stream.")
#             exit(0)

#         self.grabbed , self.frame = self.video.read()
#         self.stopped = True
#         # Creating a thread
#         self.thread = Thread(target=self.update)
#         self.thread.daemon = True
    
#     def start(self):
#         self.stopped = False
#         self.thread.start()

#     def update(self):
#         while True :
#             if self.stopped is True :
#                 break
#             self.grabbed , self.frame = self.video.read()
#         self.video.release()

#     def read(self):
#         return self.frame

#     def stop(self):
#         self.stopped = True

# # Capturing video through the WebCam. 0 represents the
# # default camera. You need to specify another number for
# # any external camera
# video_stream = VideoStream(stream=0)
# video_stream.start()

# while True:
#     if video_stream.stopped is True:
#         break
#     else:
#         # Reading the video frame
#         frame = video_stream.read()
#         # Convert the frame color-space to grayscale
#         # There are more than 150 color-space to use
#         # BGR = Blue, Green, Red
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         rects = detector(gray, 1)
#         # Get the coordinates of detected face
#         for i, face_rect in enumerate(rects):
#             left = face_rect.left() 
#             top = face_rect.top()
#             width = face_rect.right() - left
#             height = face_rect.bottom() - top

#             # Draw a rectangle around the detected face
#             # Syntax: cv2.rectangle(image, start_point, end_point,
#             #  color, thickness)
#             cv2.rectangle(frame, (left, top), (left+width, top+height),
#             (0,255,0), 2)
#             # Draw a face name with the number. 
#             # Syntax: cv2.putText(image, text, origin, font, 
#             # fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
#             # For better look, lineType = cv.LINE_AA is recommended.
#             cv2.putText(frame, f"Face {i+1}", (left - 10, top - 10), 
#             font_1, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#             '''Cropping an another frame from the detected face rectangle'''
#             frame_crop = frame[top + 10:top+height-100, left + 30: 
#             left+width - 20]
#             # Show the cropped frame
#             cv2.imshow("Cropped Frame", frame_crop)
#             # masking process
#             # Smoothing the cropped frame
#             img_blur = cv2.GaussianBlur(np.array(frame_crop),(5,5), 
#             sigmaX=1.7, sigmaY=1.7)
#             # Filterting the cropped frame through the canny filter
#             edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)
#             # Show the Canny Sample of the frame: 'frame_cropped'
#             cv2.imshow("Canny Filter", edges)

#             # Center Strip
#             edges_center = edges.T[(int(len(edges.T)/2))]
#             # 255 represents white edges. If any white edges are detected
#             # in the desired place, it will show 'Glass is Present' message
#             if 255 in edges_center:
#                 cv2.rectangle(frame, (left, top+height), (left+width, 
#                 top+height+40), (0,255,0), cv2.FILLED)
#                 cv2.putText(frame, "Glass is Present", (left+10, 
#                 top+height+20), font_1, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
#             else:
#                 cv2.rectangle(frame, (left, top+height), (left+width, 
#                 top+height+40), (0,255,0), cv2.FILLED)
#                 cv2.putText(frame, "No Glass", (left+10, top+height+20), 
#                 font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

#         # delay for processing a frame
#         delay = 0.04
#         time.sleep(delay)

#     # show result
#     cv2.imshow("Result", frame)
    
#     key = cv2.waitKey(1)
#     # Press 'q' for stop the executing of the program
#     if key == ord('q'):
#         break
# # Stop capturing video frames
# video_stream.stop()
# # closing all windows 
# # we can use any key we want just return the key we need
# cv2.destroyAllWindows()







# this is second

# import dlib
# import numpy as np
# from threading import Thread
# import cv2
# import time
# import tkinter as tk
# from tkinter import ttk

# # a cv2 font type
# font_1 = cv2.FONT_HERSHEY_SIMPLEX
# # Declare dlib frontal face detector
# detector = dlib.get_frontal_face_detector()

# # This class handles the video stream 
# # captured from the WebCam
# class VideoStream:
#     def __init__(self, stream):
#         self.video = cv2.VideoCapture(stream)
#         # Setting the FPS for the video stream
#         self.video.set(cv2.CAP_PROP_FPS, 60)

#         if self.video.isOpened() is False:
#             print("Can't access the webcam stream.")
#             exit(0)

#         self.grabbed, self.frame = self.video.read()
#         self.stopped = True
#         # Creating a thread
#         self.thread = Thread(target=self.update)
#         self.thread.daemon = True
    
#     def start(self):
#         self.stopped = False
#         self.thread.start()

#     def update(self):
#         while True:
#             if self.stopped is True:
#                 break
#             self.grabbed, self.frame = self.video.read()
#         self.video.release()

#     def read(self):
#         return self.frame

#     def stop(self):
#         self.stopped = True

# # Function to apply tint to the sunglasses region
# def apply_tint(image, tint_level):
#     tinted_image = cv2.addWeighted(image, tint_level, np.zeros_like(image), 1 - tint_level, 0)
#     return tinted_image

# # Initialize Tkinter
# root = tk.Tk()
# root.title("Sunglass Tint Adjustment")
# root.geometry("400x100")

# # Create a variable to store the tint level
# tint_level = tk.DoubleVar(value=1.0)

# # Function to update tint level when scrollbar is moved
# def update_tint_level(value):
#     tint_level.set(float(value))

# # Create a scrollbar for tint adjustment
# scrollbar = ttk.Scale(root, from_=0.0, to=1.0, variable=tint_level,
#                       orient="horizontal", command=update_tint_level)
# scrollbar.pack(fill="both", expand=True)

# # Capturing video through the WebCam. 0 represents the
# # default camera. You need to specify another number for
# # any external camera
# video_stream = VideoStream(stream=0)
# video_stream.start()

# while True:
#     if video_stream.stopped is True:
#         break
#     else:
#         frame = video_stream.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = detector(gray, 1)

#         for i, face_rect in enumerate(rects):
#             left = face_rect.left()
#             top = face_rect.top()
#             width = face_rect.right() - left
#             height = face_rect.bottom() - top

#             cv2.rectangle(frame, (left, top), (left + width, top + height),
#                           (0, 255, 0), 2)
#             cv2.putText(frame, f"Face {i + 1}", (left - 10, top - 10),
#                         font_1, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#             # Cropping an another frame from the detected face rectangle
#             frame_crop = frame[top + 10:top + height - 100, left + 30:left + width - 20]

#             # Apply tint to the sunglasses region
#             # Apply tint to the sunglasses region
#             # Cropping an another frame from the detected face rectangle
#             top_roi = top + 20
#             bottom_roi = top + height - 80
#             left_roi = left + 50
#             right_roi = left + width - 10

#             # Ensure the ROI is within the bounds of the original frame
#             if top_roi >= 0 and bottom_roi <= frame.shape[0] and left_roi >= 0 and right_roi <= frame.shape[1]:
#                 frame_crop = frame[top_roi:bottom_roi, left_roi:right_roi]

#             # Apply tint to the sunglasses region
#             tinted_sunglasses = apply_tint(frame_crop, tint_level.get())
#             if tinted_sunglasses is not None:
#                 frame[top_roi:bottom_roi, left_roi:right_roi] = tinted_sunglasses
#             else:
#                 print("ROI is out of bounds!")

            


#             # Smoothing the cropped frame
#             img_blur = cv2.GaussianBlur(np.array(frame_crop), (5, 5),
#                                         sigmaX=1.7, sigmaY=1.7)
#             edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

#             # Show the Canny Sample of the frame: 'frame_cropped'
#             cv2.imshow("Canny Filter", edges)

#             # Center Strip
#             edges_center = edges.T[(int(len(edges.T) / 2))]
#             if 255 in edges_center:
#                 cv2.rectangle(frame, (left, top + height), (left + width,
#                                                             top + height + 40), (0, 255, 0), cv2.FILLED)
#                 cv2.putText(frame, "Glass is Present", (left + 10,
#                                                         top + height + 20), font_1, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
#             else:
#                 cv2.rectangle(frame, (left, top + height), (left + width,
#                                                             top + height + 40), (0, 255, 0), cv2.FILLED)
#                 cv2.putText(frame, "No Glass", (left + 10, top + height + 20),
#                             font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

#         # delay for processing a frame
#         delay = 0.04
#         time.sleep(delay)

#     # show result
#     cv2.imshow("Result", frame)
    
#     key = cv2.waitKey(1)
#     # Press 'q' for stop the executing of the program
#     if key == ord('q'):
#         break

# # Stop capturing video frames
# video_stream.stop()
# # closing all windows 
# # we can use any key we want just return the key we need
# cv2.destroyAllWindows()
# root.destroy()  # Close the Tkinter window

# this is 3

# import dlib
# import numpy as np
# from threading import Thread
# import cv2
# import time
# import tkinter as tk
# from tkinter import ttk

# # a cv2 font type
# font_1 = cv2.FONT_HERSHEY_SIMPLEX
# # Declare dlib frontal face detector
# detector = dlib.get_frontal_face_detector()

# # This class handles the video stream 
# # captured from the WebCam
# class VideoStream:
#     def __init__(self, stream):
#         self.video = cv2.VideoCapture(stream)
#         # Setting the FPS for the video stream
#         self.video.set(cv2.CAP_PROP_FPS, 60)

#         if self.video.isOpened() is False:
#             print("Can't access the webcam stream.")
#             exit(0)

#         self.grabbed, self.frame = self.video.read()
#         self.stopped = True
#         # Creating a thread
#         self.thread = Thread(target=self.update)
#         self.thread.daemon = True
    
#     def start(self):
#         self.stopped = False
#         self.thread.start()

#     def update(self):
#         while True:
#             if self.stopped is True:
#                 break
#             self.grabbed, self.frame = self.video.read()
#         self.video.release()

#     def read(self):
#         return self.frame

#     def stop(self):
#         self.stopped = True

# # Function to create a mask for the eyes region
# def create_eyes_mask(frame, face_rect):
#     mask = np.zeros_like(frame)
#     x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
#     eyes_roi = frame[y:y + h, x:x + w]
#     mask[y:y + h, x:x + w] = eyes_roi
#     return mask

# # Function to apply tint to the sunglasses region
# def apply_tint(image, tint_level):
#     tinted_image = cv2.addWeighted(image, tint_level, np.zeros_like(image), 1 - tint_level, 0)
#     return tinted_image

# # Initialize Tkinter
# root = tk.Tk()
# root.title("Sunglass Tint Adjustment")
# root.geometry("400x100")

# # Create a variable to store the tint level
# tint_level = tk.DoubleVar(value=1.0)

# # Function to update tint level when scrollbar is moved
# def update_tint_level(value):
#     tint_level.set(float(value))

# # Create a scrollbar for tint adjustment
# scrollbar = ttk.Scale(root, from_=0.0, to=1.0, variable=tint_level,
#                       orient="horizontal", command=update_tint_level)
# scrollbar.pack(fill="both", expand=True)

# # Capturing video through the WebCam. 0 represents the
# # default camera. You need to specify another number for
# # any external camera
# video_stream = VideoStream(stream=0)
# video_stream.start()

# while True:
#     if video_stream.stopped is True:
#         break
#     else:
#         frame = video_stream.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = detector(gray, 1)

#         for i, face_rect in enumerate(rects):
#             left = face_rect.left()
#             top = face_rect.top()
#             width = face_rect.right() - left
#             height = face_rect.bottom() - top

#             cv2.rectangle(frame, (left, top), (left + width, top + height),
#                           (0, 255, 0), 2)
#             cv2.putText(frame, f"Face {i + 1}", (left - 10, top - 10),
#                         font_1, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#             # Cropping an another frame from the detected face rectangle
#             top_roi = top + 10
#             bottom_roi = top + height - 100
#             left_roi = left + 30
#             right_roi = left + width - 20

#             # Ensure the ROI is within the bounds of the original frame
#             if top_roi >= 0 and bottom_roi <= frame.shape[0] and left_roi >= 0 and right_roi <= frame.shape[1]:
#                 frame_crop = frame[top_roi:bottom_roi, left_roi:right_roi]

#                 # Apply tint to the sunglasses region
#                 tinted_sunglasses = apply_tint(frame_crop, tint_level.get())
#                 if tinted_sunglasses is not None:
#                     frame[top_roi:bottom_roi, left_roi:right_roi] = tinted_sunglasses

#                 # Create a mask for the eyes region with full color
#                 eyes_mask = create_eyes_mask(frame, face_rect)

#                 # Show the eyes mask in a separate window
#                 cv2.imshow("Eyes Mask", eyes_mask)

#         delay = 0.04
#         time.sleep(delay)

#     # Show the result
#     cv2.imshow("Result", frame)
    
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# # Stop capturing video frames
# video_stream.stop()
# # closing all windows 
# # we can use any key we want just return the key we need
# cv2.destroyAllWindows()
# root.destroy()  # Close the Tkinter window





# this is the 4

# import dlib
# import numpy as np
# from threading import Thread
# import cv2
# import time
# import tkinter as tk
# from tkinter import ttk

# # a cv2 font type
# font_1 = cv2.FONT_HERSHEY_SIMPLEX
# # Declare dlib frontal face detector
# detector = dlib.get_frontal_face_detector()

# # This class handles the video stream 
# # captured from the WebCam
# class VideoStream:
#     def __init__(self, stream):
#         self.video = cv2.VideoCapture(stream)
#         # Setting the FPS for the video stream
#         self.video.set(cv2.CAP_PROP_FPS, 60)

#         if self.video.isOpened() is False:
#             print("Can't access the webcam stream.")
#             exit(0)

#         self.grabbed, self.frame = self.video.read()
#         self.stopped = True
#         # Creating a thread
#         self.thread = Thread(target=self.update)
#         self.thread.daemon = True
    
#     def start(self):
#         self.stopped = False
#         self.thread.start()

#     def update(self):
#         while True:
#             if self.stopped is True:
#                 break
#             self.grabbed, self.frame = self.video.read()
#         self.video.release()

#     def read(self):
#         return self.frame

#     def stop(self):
#         self.stopped = True

# # Function to apply tint to the sunglasses region
# def apply_tint(image, tint_level):
#     tinted_image = cv2.addWeighted(image, tint_level, np.zeros_like(image), 1 - tint_level, 0)
#     return tinted_image

# # Initialize Tkinter
# root = tk.Tk()
# root.title("Sunglass Tint Adjustment")
# root.geometry("400x100")

# # Create a variable to store the tint level
# tint_level = tk.DoubleVar(value=1.0)

# # Function to update tint level when scrollbar is moved
# def update_tint_level(value):
#     tint_level.set(float(value))

# # Create a scrollbar for tint adjustment
# scrollbar = ttk.Scale(root, from_=0.0, to=1.0, variable=tint_level,
#                       orient="horizontal", command=update_tint_level)
# scrollbar.pack(fill="both", expand=True)

# # Capturing video through the WebCam. 0 represents the
# # default camera. You need to specify another number for
# # any external camera
# video_stream = VideoStream(stream=0)
# video_stream.start()

# while True:
#     if video_stream.stopped is True:
#         break
#     else:
#         frame = video_stream.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = detector(gray, 1)

#         for i, face_rect in enumerate(rects):
#             left = face_rect.left()
#             top = face_rect.top()
#             width = face_rect.right() - left
#             height = face_rect.bottom() - top

#             cv2.rectangle(frame, (left, top), (left + width, top + height),
#                           (0, 255, 0), 2)
#             cv2.putText(frame, f"Face {i + 1}", (left - 10, top - 10),
#                         font_1, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#             # Cropping an another frame from the detected face rectangle
#             top_roi = top + 10
#             bottom_roi = top + height - 100
#             left_roi = left + 30
#             right_roi = left + width - 20

#             # Ensure the ROI is within the bounds of the original frame
#             if top_roi >= 0 and bottom_roi <= frame.shape[0] and left_roi >= 0 and right_roi <= frame.shape[1]:
#                 frame_crop = frame[top_roi:bottom_roi, left_roi:right_roi]

#                 # Apply tint to the sunglasses region
#                 tinted_sunglasses = apply_tint(frame_crop, tint_level.get())
#                 if tinted_sunglasses is not None:
#                     frame[top_roi:bottom_roi, left_roi:right_roi] = tinted_sunglasses

#                     # Create a mask for the eyes region with full color
#                     eyes_mask = np.zeros_like(frame)
#                     eyes_roi = frame[top_roi:bottom_roi, left_roi:right_roi]
#                     eyes_mask[top_roi:bottom_roi, left_roi:right_roi] = eyes_roi

#                     # Show the eyes mask in a separate window
#                     cv2.imshow("Eyes Mask", eyes_mask)

#                 # Smoothing the cropped frame
#                 img_blur = cv2.GaussianBlur(np.array(frame_crop), (5, 5),
#                                             sigmaX=1.7, sigmaY=1.7)
#                 edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

#                 # Show the Canny Sample of the frame: 'frame_cropped'
#                 cv2.imshow("Canny Filter", edges)

#                 # Center Strip
#                 edges_center = edges.T[(int(len(edges.T) / 2))]
#                 if 255 in edges_center:
#                     cv2.rectangle(frame, (left, top + height), (left + width,
#                                                                 top + height + 40), (0, 255, 0), cv2.FILLED)
#                     cv2.putText(frame, "Glass is Present", (left + 10,
#                                                             top + height + 20), font_1, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
#                 else:
#                     cv2.rectangle(frame, (left, top + height), (left + width,
#                                                                 top + height + 40), (0, 255, 0), cv2.FILLED)
#                     cv2.putText(frame, "No Glass", (left + 10, top + height + 20),
#                                 font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

#         delay = 0.04
#         time.sleep(delay)

#     # Show the result
#     cv2.imshow("Result", frame)
    
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# # Stop capturing video frames
# video_stream.stop()
# # closing all windows 
# # we can use any key we want just return the key we need
# cv2.destroyAllWindows()
# root.destroy()  # Close the Tkinter window



# this is the 5 

# import dlib
# import numpy as np
# from threading import Thread
# import cv2
# import time
# import tkinter as tk
# from tkinter import ttk

# # a cv2 font type
# font_1 = cv2.FONT_HERSHEY_SIMPLEX
# # Declare dlib frontal face detector
# detector = dlib.get_frontal_face_detector()

# # This class handles the video stream 
# # captured from the WebCam
# class VideoStream:
#     def __init__(self, stream):
#         self.video = cv2.VideoCapture(stream)
#         # Setting the FPS for the video stream
#         self.video.set(cv2.CAP_PROP_FPS, 60)

#         if self.video.isOpened() is False:
#             print("Can't access the webcam stream.")
#             exit(0)

#         self.grabbed, self.frame = self.video.read()
#         self.stopped = True
#         # Creating a thread
#         self.thread = Thread(target=self.update)
#         self.thread.daemon = True
    
#     def start(self):
#         self.stopped = False
#         self.thread.start()

#     def update(self):
#         while True:
#             if self.stopped is True:
#                 break
#             self.grabbed, self.frame = self.video.read()
#         self.video.release()

#     def read(self):
#         return self.frame

#     def stop(self):
#         self.stopped = True

# # Function to create a mask for the eyes region
# def create_eyes_mask(frame, face_rect):
#     mask = np.zeros_like(frame)
#     x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
#     eyes_roi = frame[y:y + h, x:x + w]
#     mask[y:y + h, x:x + w] = eyes_roi
#     return mask

# # Function to apply tint to the sunglasses region
# def apply_tint(image, tint_level):
#     tinted_image = cv2.addWeighted(image, tint_level, np.zeros_like(image), 1 - tint_level, 0)
#     return tinted_image

# # Initialize Tkinter
# root = tk.Tk()
# root.title("Sunglass Tint Adjustment")
# root.geometry("400x100")

# # Create a variable to store the tint level
# tint_level = tk.DoubleVar(value=1.0)

# # Function to update tint level when scrollbar is moved
# def update_tint_level(value):
#     tint_level.set(float(value))

# # Create a scrollbar for tint adjustment
# scrollbar = ttk.Scale(root, from_=0.0, to=1.0, variable=tint_level,
#                       orient="horizontal", command=update_tint_level)
# scrollbar.pack(fill="both", expand=True)

# # Capturing video through the WebCam. 0 represents the
# # default camera. You need to specify another number for
# # any external camera
# video_stream = VideoStream(stream=0)
# video_stream.start()

# while True:
#     if video_stream.stopped is True:
#         break
#     else:
#         frame = video_stream.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = detector(gray, 1)

#         for i, face_rect in enumerate(rects):
#             left = face_rect.left()
#             top = face_rect.top()
#             width = face_rect.right() - left
#             height = face_rect.bottom() - top

#             cv2.rectangle(frame, (left, top), (left + width, top + height),
#                           (0, 255, 0), 2)
#             cv2.putText(frame, f"Face {i + 1}", (left - 10, top - 10),
#                         font_1, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#             # Cropping an another frame from the detected face rectangle
#             top_roi = top + 10
#             bottom_roi = top + height - 100
#             left_roi = left + 30
#             right_roi = left + width - 20

#             # Ensure the ROI is within the bounds of the original frame
#             if top_roi >= 0 and bottom_roi <= frame.shape[0] and left_roi >= 0 and right_roi <= frame.shape[1]:
#                 frame_crop = frame[top_roi:bottom_roi, left_roi:right_roi]

#                 # Create a mask for the eyes region with full color
#                 eyes_mask = create_eyes_mask(frame, face_rect)

#                 # Apply tint to the sunglasses region
#                 tinted_sunglasses = apply_tint(frame_crop, tint_level.get())
#                 if tinted_sunglasses is not None:
#                     frame[top_roi:bottom_roi, left_roi:right_roi] = tinted_sunglasses

#                     # Create a mask for the tinted eyes region
#                     tinted_eyes_mask = apply_tint(eyes_mask, tint_level.get())

#                     # Show the tinted eyes mask in a separate window
#                     cv2.imshow("Tinted Eyes Mask", tinted_eyes_mask)

#                 # Smoothing the cropped frame
#                 img_blur = cv2.GaussianBlur(np.array(frame_crop), (5, 5),
#                                             sigmaX=1.7, sigmaY=1.7)
#                 edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

#                 # Show the Canny Sample of the frame: 'frame_cropped'
#                 cv2.imshow("Canny Filter", edges)

#                 # Center Strip
#                 edges_center = edges.T[(int(len(edges.T) / 2))]
#                 if 255 in edges_center:
#                     cv2.rectangle(frame, (left, top + height), (left + width,
#                                                                 top + height + 40), (0, 255, 0), cv2.FILLED)
#                     cv2.putText(frame, "Glass is Present", (left + 10,
#                                                             top + height + 20), font_1, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
#                 else:
#                     cv2.rectangle(frame, (left, top + height), (left + width,
#                                                                 top + height + 40), (0, 255, 0), cv2.FILLED)
#                     cv2.putText(frame, "No Glass", (left + 10, top + height + 20),
#                                 font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

#         delay = 0.04
#         time.sleep(delay)

#     # Show the result
#     cv2.imshow("Result", frame)
    
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# # Stop capturing video frames
# video_stream.stop()
# # closing all windows 
# # we can use any key we want just return the key we need
# cv2.destroyAllWindows()
# root.destroy()  # Close the Tkinter window




# this is 6

# import dlib
# import numpy as np
# from threading import Thread
# import cv2
# import time
# import tkinter as tk

# # a cv2 font type
# font_1 = cv2.FONT_HERSHEY_SIMPLEX
# # Declare dlib frontal face detector
# detector = dlib.get_frontal_face_detector()

# # This class handles the video stream 
# # captured from the WebCam
# class VideoStream:
#     def __init__(self, stream):
#         self.video = cv2.VideoCapture(stream)
#         # Setting the FPS for the video stream
#         self.video.set(cv2.CAP_PROP_FPS, 60)

#         if self.video.isOpened() is False:
#             print("Can't access the webcam stream.")
#             exit(0)

#         self.grabbed, self.frame = self.video.read()
#         self.stopped = True
#         # Creating a thread
#         self.thread = Thread(target=self.update)
#         self.thread.daemon = True
    
#     def start(self):
#         self.stopped = False
#         self.thread.start()

#     def update(self):
#         while True:
#             if self.stopped is True:
#                 break
#             self.grabbed, self.frame = self.video.read()
#         self.video.release()

#     def read(self):
#         return self.frame

#     def stop(self):
#         self.stopped = True

# # Function to create a mask for the eyes region
# def create_eyes_mask(frame, face_rect):
#     mask = np.zeros_like(frame)
#     x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
#     eyes_roi = frame[y:y + h, x:x + w]
#     mask[y:y + h, x:x + w] = eyes_roi
#     return mask

# # Function to apply tint to the sunglasses region
# def apply_tint(image, tint_level):
#     tinted_image = cv2.addWeighted(image, tint_level, np.zeros_like(image), 1 - tint_level, 0)
#     return tinted_image

# # Initialize tint level
# tint_level = 1.0

# # Capturing video through the WebCam. 0 represents the
# # default camera. You need to specify another number for
# # any external camera
# video_stream = VideoStream(stream=0)
# video_stream.start()

# while True:
#     if video_stream.stopped is True:
#         break
#     else:
#         frame = video_stream.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = detector(gray, 1)

#         for i, face_rect in enumerate(rects):
#             left = face_rect.left()
#             top = face_rect.top()
#             width = face_rect.right() - left
#             height = face_rect.bottom() - top

#             cv2.rectangle(frame, (left, top), (left + width, top + height),
#                           (0, 255, 0), 2)
#             cv2.putText(frame, f"Face {i + 1}", (left - 10, top - 10),
#                         font_1, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#             # Cropping another frame from the detected face rectangle
#             top_roi = top + 10
#             bottom_roi = top + height - 100
#             left_roi = left + 30
#             right_roi = left + width - 20

#             # Ensure the ROI is within the bounds of the original frame
#             if top_roi >= 0 and bottom_roi <= frame.shape[0] and left_roi >= 0 and right_roi <= frame.shape[1]:
#                 frame_crop = frame[top_roi:bottom_roi, left_roi:right_roi]

#                 # Create a mask for the eyes region with full color
#                 eyes_mask = create_eyes_mask(frame, face_rect)

#                 # Apply tint to the sunglasses region
#                 tinted_sunglasses = apply_tint(frame_crop, tint_level)
#                 if tinted_sunglasses is not None:
#                     frame[top_roi:bottom_roi, left_roi:right_roi] = tinted_sunglasses

#                     # Create a mask for the tinted eyes region
#                     tinted_eyes_mask = apply_tint(eyes_mask, tint_level)

#                     # Show the tinted eyes mask in a separate window
#                     cv2.imshow("Tinted Eyes Mask", tinted_eyes_mask)

#                 # Smoothing the cropped frame
#                 img_blur = cv2.GaussianBlur(np.array(frame_crop), (5, 5),
#                                             sigmaX=1.7, sigmaY=1.7)
#                 edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

#                 # Show the Canny Sample of the frame: 'frame_cropped'
#                 cv2.imshow("Canny Filter", edges)

#                 # Center Strip
#                 edges_center = edges.T[(int(len(edges.T) / 2))]
#                 if 255 in edges_center:
#                     cv2.rectangle(frame, (left, top + height), (left + width,
#                                                                 top + height + 40), (0, 255, 0), cv2.FILLED)
#                     cv2.putText(frame, "Glass is Present", (left + 10,
#                                                             top + height + 20), font_1, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
#                 else:
#                     cv2.rectangle(frame, (left, top + height), (left + width,
#                                                                 top + height + 40), (0, 255, 0), cv2.FILLED)
#                     cv2.putText(frame, "No Glass", (left + 10, top + height + 20),
#                                 font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

#         delay = 0.04
#         time.sleep(delay)

#     # Show the result
#     cv2.imshow("Result", frame)

#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#     elif key == cv2.EVENT_FLAG_ALTKEY:
#         # Adjust tint using PgUp (increase tint) and PgDn (decrease tint) keys
#         if key == cv2.EVENT_PAGEDOWN:
#             tint_level = max(0.0, tint_level - 0.1)
#         elif key == cv2.EVENT_PAGEUP:
#             tint_level = min(1.0, tint_level + 0.1)

# # Stop capturing video frames
# video_stream.stop()
# # closing all windows
# cv2.destroyAllWindows()



# this is 7

import dlib
import numpy as np
from threading import Thread
import cv2
import time

# a cv2 font type
font_1 = cv2.FONT_HERSHEY_SIMPLEX
# Declare dlib frontal face detector
detector = dlib.get_frontal_face_detector()

# This class handles the video stream
# captured from the WebCam
class VideoStream:
    def __init__(self, stream):
        self.video = cv2.VideoCapture(stream)
        # Setting the FPS for the video stream
        self.video.set(cv2.CAP_PROP_FPS, 60)

        if self.video.isOpened() is False:
            print("Can't access the webcam stream.")
            exit(0)

        self.grabbed, self.frame = self.video.read()
        self.stopped = True
        # Creating a thread
        self.thread = Thread(target=self.update)
        self.thread.daemon = True

    def start(self):
        self.stopped = False
        self.thread.start()

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.video.read()
        self.video.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Function to create a mask for the eyes region
def create_eyes_mask(frame, face_rect):
    mask = np.zeros_like(frame)
    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    eyes_roi = frame[y:y + h, x:x + w]
    mask[y:y + h, x:x + w] = eyes_roi
    return mask

# Function to apply tint to the sunglasses region
def apply_tint(image, tint_level):
    tinted_image = cv2.addWeighted(image, tint_level, np.zeros_like(image), 1 - tint_level, 0)
    return tinted_image

# Initialize tint level
tint_level = 1.0

# Capturing video through the WebCam. 0 represents the
# default camera. You need to specify another number for
# any external camera
video_stream = VideoStream(stream=0)
video_stream.start()

while True:
    if video_stream.stopped is True:
        break
    else:
        frame = video_stream.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for i, face_rect in enumerate(rects):
            left = face_rect.left()
            top = face_rect.top()
            width = face_rect.right() - left
            height = face_rect.bottom() - top

            cv2.rectangle(frame, (left, top), (left + width, top + height),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"Face {i + 1}", (left - 10, top - 10),
                        font_1, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Cropping another frame from the detected face rectangle
            top_roi = top + 10
            bottom_roi = top + height - 100
            left_roi = left + 30
            right_roi = left + width - 20

            # Ensure the ROI is within the bounds of the original frame
            if top_roi >= 0 and bottom_roi <= frame.shape[0] and left_roi >= 0 and right_roi <= frame.shape[1]:
                frame_crop = frame[top_roi:bottom_roi, left_roi:right_roi]

                # Create a mask for the eyes region with full color
                eyes_mask = create_eyes_mask(frame, face_rect)

                # Apply tint to the sunglasses region
                tinted_sunglasses = apply_tint(frame_crop, tint_level)
                if tinted_sunglasses is not None:
                    frame[top_roi:bottom_roi, left_roi:right_roi] = tinted_sunglasses

                    # Create a mask for the tinted eyes region
                    tinted_eyes_mask = apply_tint(eyes_mask, tint_level)

                    # Show the tinted eyes mask in a separate window
                    cv2.imshow("Tinted Eyes Mask", tinted_eyes_mask)

                # Smoothing the cropped frame
                img_blur = cv2.GaussianBlur(np.array(frame_crop), (5, 5),
                                            sigmaX=1.7, sigmaY=1.7)
                edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

                # Show the Canny Sample of the frame: 'frame_cropped'
                cv2.imshow("Canny Filter", edges)

                # Center Strip
                edges_center = edges.T[(int(len(edges.T) / 2))]
                if 255 in edges_center:
                    cv2.rectangle(frame, (left, top + height), (left + width,
                                                                top + height + 40), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, "Glass is Present", (left + 10,
                                                            top + height + 20), font_1, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (left, top + height), (left + width,
                                                                top + height + 40), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, "No Glass", (left + 10, top + height + 20),
                                font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        delay = 0.04
        time.sleep(delay)

    # Show the result
    cv2.imshow("Result", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == 0x22:  # PgDn key
        tint_level = max(0.0, tint_level - 0.1)
    elif key == 0x21:  # PgUp key
        tint_level = min(1.0, tint_level + 0.1)

# Stop capturing video frames
video_stream.stop()
# closing all windows
cv2.destroyAllWindows()


# this is not working fghgfgh