from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
import urllib.request
from datetime import datetime

class ESP32CamApp(App):
    def build(self):
        self.url = 'http://192.168.137.69/cam-hi.jpg'  # Replace with your ESP32Cam stream URL
        
        self.img_widget = Image()
        
        # Create the layout for controls
        control_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        
        # Create the start and stop buttons
        self.start_button = Button(text="Start Recording")
        self.stop_button = Button(text="Stop Recording")
        self.start_button.bind(on_press=self.start_recording)
        self.stop_button.bind(on_press=self.stop_recording)
        self.stop_button.disabled = True
        
        # Create a frame rate label and slider
        self.frame_rate_label = Label(text="Frame Rate: 30")
        self.frame_rate_slider = Slider(min=1, max=60, value=30, step=1, orientation='horizontal')
        self.frame_rate_slider.bind(value=self.update_frame_rate)
        
        # Add the buttons and slider to the control layout
        control_layout.add_widget(self.start_button)
        control_layout.add_widget(self.stop_button)
        control_layout.add_widget(self.frame_rate_label)
        control_layout.add_widget(self.frame_rate_slider)
        
        # Create the main layout and add the image widget and control layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img_widget)
        layout.add_widget(control_layout)
        
        # Initialize recording-related variables
        self.recording = False
        self.video_writer = None
        
        # Set the frame rate and schedule the stream update
        self.frame_rate = 30
        Clock.schedule_interval(self.update_stream, 1.0 / self.frame_rate)

        return layout

    def update_stream(self, dt):
        # Update the stream if not recording
        if self.recording:
            try:
                img_response = urllib.request.urlopen(self.url)
                img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
                img = cv2.imdecode(img_np, -1)
                
                # Update the texture of the image widget
                self.img_widget.texture = self._convert_opencv_image(img)
                
                # If recording, write the frame to the video file
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(img)
            except Exception as e:
                print(f"Error updating stream: {e}")

    def start_recording(self, instance):
        # Check if the image widget's texture is initialized
        if self.img_widget.texture is None:
            print("Cannot start recording: Texture not initialized.")
            return
        
        # Set recording flag and disable the start button
        self.recording = True
        self.start_button.disabled = True
        self.stop_button.disabled = False
        
        # Create a video writer object for recording
        video_filename = f"stream_{datetime.now().strftime('%Y%m%d%H%M%S')}.avi"
        
        # Get the frame size from the texture
        frame_size = (self.img_widget.texture.size[0], self.img_widget.texture.size[1])
        
        # Define the codec and create the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(video_filename, fourcc, self.frame_rate, frame_size)

    def stop_recording(self, instance):
        # Stop recording and enable the start button
        self.recording = False
        self.start_button.disabled = False
        self.stop_button.disabled = True
        
        # Release the video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def update_frame_rate(self, instance, value):
        # Update the frame rate and reschedule the stream update
        self.frame_rate = int(value)
        self.frame_rate_label.text = f"Frame Rate: {self.frame_rate}"
        Clock.unschedule(self.update_stream)
        Clock.schedule_interval(self.update_stream, 1.0 / self.frame_rate)

    def _convert_opencv_image(self, img):
        # Convert an OpenCV image to a Kivy texture
        buf = img.tobytes()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

if __name__ == '__main__':
    ESP32CamApp().run()

