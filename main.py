import kivy
from tkinter import Image
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
import speech_recognition as sr
from kivy.uix.camera import Camera
import os
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle
import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import model_from_json  
from keras.preprocessing import image
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.lang import Builder
from PIL import Image
from kivy.uix.gridlayout import GridLayout
from kivy.core.text import LabelBase
from kivy.uix.image import Image,AsyncImage



word_dict = {0:'Ano',1:'Bakit',2:'Bukas',3:'Ingat',4:'Mahal-Kita',5:'Paalam',6:'Paano',7:'Paumanhin',8:'SAAN-KA-NAKATIRA',9:'Walang-Anuman'}
model = keras.models.load_model(r"best_model_dataflair3.h5")

    
background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

Builder.load_string('''
<CustomLayout>
    canvas.before:
        Color:
            rgba: 0, 1, 0, 1
        Rectangle:
            pos: self.pos
            size: self.size
<RootWidget>
    CustomLayout:
        AsyncImage:
            source: 'logo.png'
            size_hint: 1, .5
            pos_hint: {'center_x':.5, 'center_y': .5}''')

class RootWidget(BoxLayout):
    pass

class CustomLayout(FloatLayout):
    pass
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Now...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio, language='tl-PH')
        print("Text: ", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


class HomeScreen(Screen):
    
    def __init__(self, **kwargs):
        super(HomeScreen, self).__init__(**kwargs)
        

        layout = BoxLayout(orientation='vertical')
        self.add_widget(layout)

        LabelBase.register(name='GeneralSans-Bold',fn_regular='GeneralSans-Bold.ttf')
        label = Label(text='FSL RECOGNITION', size_hint=(1, 0.95),pos_hint={'x':-0.01,'y':1.3}, color=(0,0.5,0.1,1), font_name= "GeneralSans-Bold", font_size ='23sp')
        
        
        
        layout.add_widget(label)
        
    
        LabelBase.register(name='GeneralSans-Bold',fn_regular='GeneralSans-Bold.ttf')
        button = Button(text='MUTE / DEAF',background_color = (0,1,0.0,1), size_hint={0.47,0.12},pos_hint={'x':0.25,'y':0.2}, on_press=self.go_to_mutedeaf_screen)
        layout.add_widget(button)
    
       

        button = Button(text='HEAR / TALK',background_color = (0,1,0.0,1),size_hint={0.47,0.12},pos_hint={'x':0.25,'y':0.2}, on_press=self.go_to_heartalk_screen)
        layout.add_widget(button)
        LabelBase.register(name='GeneralSans-Bold',fn_regular='GeneralSans-Bold.ttf')


    def go_to_mutedeaf_screen(self, *args):
        self.manager.current = 'mutedeaf'

    def go_to_heartalk_screen(self, *args):
        self.manager.current = 'heartalk'

    

class MuteDeafScreen(Screen):
    def __init__(self, **kwargs):
        super(MuteDeafScreen, self).__init__(**kwargs)

        layout = BoxLayout(orientation='vertical')
        self.add_widget(layout)

        label = Label(text=' ', size_hint=(1, 0.9))
        layout.add_widget(label)

        result_label = Label(text='', size_hint=(1, 0.9))
        layout.add_widget(result_label)

        button = Button(text='Start Recording',background_color = (0,1,0.0,1),size_hint={0.47,0.2},pos_hint={'x':0.25,'y':0.2}, on_press=self.start_recording)
        layout.add_widget(button)
        

        button = Button(text='Back',background_color = (0,1,0.0,1), size_hint={0.47,0.2},pos_hint={'x':0.25,'y':0.2}, on_press=self.go_to_home_screen)
        layout.add_widget(button)

        self.result_label = result_label

    def go_to_home_screen(self, *args):
        self.manager.current = 'home'

    def start_recording(self, *args):
        text = speech_to_text()
        self.result_label.text = text

class HearTalkScreen(Screen):
    def __init__(self, **kwargs):
        super(HearTalkScreen, self).__init__(**kwargs)

        layout = BoxLayout(orientation='vertical')
        self.add_widget(layout)

        self.sign_lbl = Label(text='', font_size=30)
        layout.add_widget(self.sign_lbl)
        

        self.capture_btn = Button(text='Start Video', size_hint={0.47,0.12},background_color = (0,1,0.0,1),pos_hint={'x':0.25,'y':0.2}, on_press=self.start_video)
        layout.add_widget(self.capture_btn)
        
        #self.start_video =  BoxLayout(orientation='vertical')
        #layout.add_widget(self.start_video)

        button = Button(text='Back', size_hint={0.47,0.12},background_color = (0,1,0.0,1),pos_hint={'x':0.25,'y':0.2}, on_press=self.go_to_home_screen)
        layout.add_widget(button)

    def go_to_home_screen(self, *args):
        self.manager.current = 'home'

        
    def start_video(self, instance):
    # Define the codec and create VideoWriter object
        #self.camera.play = True
        #self.camera.recording = True
        
        #self.camera.play = True
        #self.camera.filename = os.path.join(os.getcwd(), "test.avi")
        #self.camera.recording = True
        #self.capture_btn.disabled = True

        def cal_accum_avg(frame, accumulated_weight):

            global background
            
            if background is None:
                background = frame.copy().astype("float")
                return None

            cv2.accumulateWeighted(frame, background, accumulated_weight)

        def segment_hand(frame, threshold=25):
            global background
            
            diff = cv2.absdiff(background.astype("uint8"), frame)

            
            _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            #Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
            contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If length of contours list = 0, means we didn't get any contours...
            if len(contours) == 0:
                return None
            else:
                # The largest external contour should be the hand 
                hand_segment_max_cont = max(contours, key=cv2.contourArea)
                
                # Returning the hand segment(max contour) and the thresholded image of hand...
                return (thresholded, hand_segment_max_cont)

        cam = cv2.VideoCapture(0) 
        num_frames =0

        while True:
            ret, frame = cam.read()

            # filpping the frame to prevent inverted image of captured frame...
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()

            # ROI from the frame
            roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

            if num_frames < 70:
                
                cal_accum_avg(gray_frame, accumulated_weight)
                
                cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
            else: 
                # segmenting the hand region
                hand = segment_hand(gray_frame)
                

                # Checking if we are able to detect the hand...
                if hand is not None:
                    
                    thresholded, hand_segment = hand

                    # Drawing contours around hand segment
                    cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)
                    
                    cv2.imshow("Thesholded Hand Image", thresholded)
                    
                    thresholded = cv2.resize(thresholded, (64, 64))
                    thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                    thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))
                    
                    pred = model.predict(thresholded)
                    cv2.putText(frame_copy, word_dict[np.argmax(pred)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
            # Draw ROI on frame_copy
            cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)

            # incrementing the number of frames for tracking
            num_frames += 1

            # Display the frame with segmented hand
            cv2.putText(frame_copy, "Hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
            cv2.imshow("Sign Detection", frame_copy)

            # Close windows with Esc
            k = cv2.waitKey(1) & 0xFF

            if k == 27:
                break

        # Release the camera and destroy all the windows
        cam.release()



        
class MainApp(MDApp):
    def build(self):
        
        Window.clearcolor = 0.890,0.885,0.890,1

        sm = ScreenManager()
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(MuteDeafScreen(name='mutedeaf'))
        sm.add_widget(HearTalkScreen(name='heartalk'))
    

        return sm
    
    

    
if __name__ == '__main__':
    MainApp().run()
    
