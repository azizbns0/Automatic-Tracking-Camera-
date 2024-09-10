import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from imageai.Detection import ObjectDetection

class ESP32CamApp(App):
    def build(self):
        # Create a vertical layout
        layout = BoxLayout(orientation='vertical')
        
        # Create an Image widget to display frames from the stream
        self.image_widget = Image()
        layout.add_widget(self.image_widget)
        
        # Create a horizontal layout for the buttons
        button_layout = BoxLayout(size_hint_y=0.1)
        
        # Create start and stop buttons
        self.start_button = Button(text='Start Streaming')
        self.stop_button = Button(text='Stop Streaming', disabled=True)
        
        # Add buttons to the layout
        button_layout.add_widget(self.start_button)
        button_layout.add_widget(self.stop_button)
        
        # Add button layout to the main layout
        layout.add_widget(button_layout)
        
        # Define the ESP32-CAM stream URL (update this with your ESP32-CAM's stream URL)
        self.stream_url = 'http://192.168.137.76/cam-hi.jpg'  # Update with your ESP32-CAM stream URL
        
        # Initialize streaming flag
        self.is_streaming = False
        
        # Initialize the object detection model
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath("yolo.h5")  # Provide the correct path to the YOLOv3 model file
        self.detector.loadModel()
        
        # Bind button events to the appropriate methods
        self.start_button.bind(on_press=self.start_streaming)
        self.stop_button.bind(on_press=self.stop_streaming)
        
        return layout
    
    def start_streaming(self, instance):
        # Start streaming video at 30 frames per second
        if not self.is_streaming:
            self.is_streaming = True
            self.start_button.disabled = True
            self.stop_button.disabled = False
            Clock.schedule_interval(self.update_image, 1.0 / 30)  # Update image at 30 FPS
    
    def stop_streaming(self, instance):
        # Stop streaming video
        if self.is_streaming:
            self.is_streaming = False
            self.start_button.disabled = False
            self.stop_button.disabled = True
            Clock.unschedule(self.update_image)
    
    def update_image(self, dt):
        # Capture video from the ESP32-CAM stream
        cap = cv2.VideoCapture(self.stream_url)
        
        # Check if the video stream is open
        if not cap.isOpened():
            print(f"Failed to open stream at {self.stream_url}")
            return
        
        # Read a frame from the stream
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from the stream")
            cap.release()
            return
        
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform object detection on the frame
        detections = self.detector.detectObjectsFromImage(input_type="array", input_image=frame_rgb, output_type="array")
        
        # Draw bounding boxes and labels around detected objects
        for detection in detections:
            label, probability, box_points = detection
            x1, y1, x2, y2 = map(int, box_points)
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_rgb, f"{label} ({probability:.2f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check if a person is detected
        person_detected = any(detection[0] == "person" for detection in detections)
        
        # Add text to the frame to indicate if a person is detected
        if person_detected:
            cv2.putText(frame_rgb, "Person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame_rgb, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Ensure the Image widget's texture exists
        if self.image_widget.texture is None:
            self.image_widget.texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
        
        # Update the Image widget's texture with the frame data
        self.image_widget.texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        
        # Properly release the video capture object
        cap.release()

# Start the application
if __name__ == "__main__":
    ESP32CamApp().run()
