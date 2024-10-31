import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import urllib.request  # Library to open and read URLs
import numpy as np  # Library for numerical operations


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

        # Define window name for displaying the camera feed
        self.winName = 'ESP32 CAMERA'
        cv2.namedWindow(self.winName, cv2.WINDOW_AUTOSIZE)

        # Load class names for object detection
        self.classNames = []
        classFile = 'hsin/coco.names'
        with open(classFile, 'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

        # Define paths to pre-trained model files
        configPath = 'hsin/ssd_mobileself.net_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'hsin/frozen_inference_graph.pb'

        # Load the pre-trained object detection model
        self.net = cv2.dnn.DetectionModel(weightsPath, configPath)
        self.net.setInputSize(320, 320)  # Set the input size for the model
        self.net.setInputScale(1.0 / 127.5)  # Normalize the input image
        self.net.setInputMean((127.5, 127.5, 127.5))  # Set the mean value for each channel
        self.net.setInputSwapRB(True)  # Swap Red and Blue channels for BGR images

        
        # Initialize streaming flag
        self.is_streaming = False
        
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
            Clock.schedule_interval(self.update_image, 1.0 /60)  # Update image at 30 FPS
    
    def stop_streaming(self, instance):
        # Stop streaming video
        if self.is_streaming:
            self.is_streaming = False
            self.start_button.disabled = False
            self.stop_button.disabled = True
            Clock.unschedule(self.update_image)
    
    def update_image(self, dt):
        # Main loop for processing video frames
        while True:
            # Open the URL and read the image data
            imgResponse = urllib.request.urlopen(self.stream_url)
            imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)  # Decode the image data

            # Get the center coordinates of the image frame
            center_x = img.shape[1] // 2
            #center_y = img.shape[0] // 2

            # Use the model to detect objects in the image
            classIds, confs, bbox = self.net.detect(img, confThreshold=0.5)

            # Check if person class (class ID 1) is detected
            if 1 in classIds:
                # Get the index of the person class in the detections
                lista = classIds.tolist()
                d = lista.index(1)

                # Loop through each person detection
                for classId, confidence, box in zip(classIds[d].flatten(), confs[d].flatten(), bbox):
                    # Draw a rectangle around the detected person
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)

                    # Put a label on the rectangle with the class name (person)
                    cv2.putText(img, self.classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    # Calculate the center coordinates of the bounding box
                    box_center_x = box[0] + (box[2] // 2)

                    # Calculate the error (distance) between the center of the frame and the person
                    error_x = center_x - box_center_x

            # Display the image with detections
            cv2.imshow(self.winName, img)

            # Wait for 'Esc' key to exit the program
            tecla = cv2.waitKey(5) & 0xFF
            if tecla == 27:
                break

if __name__ == '__main__':
    ESP32CamApp().run()






    
