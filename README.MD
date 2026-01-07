✨ PresentationPal

Gesture-Controlled Presentation, Virtual Board, and Image Zoom System

🚀 Overview

PresentationPal is a vision-based, gesture-controlled system that enables users to interact with presentations, draw on a virtual board, and zoom images using hand gestures instead of traditional input devices like mouse or keyboard.

Built using Python, OpenCV, MediaPipe, and Django, PresentationPal demonstrates how computer vision can be used for Human–Computer Interaction (HCI) by translating real-time hand movements into meaningful system actions.

🎯 What Makes PresentationPal Unique?

Touchless Interaction: Control slides, draw, and zoom using hand gestures.

Computer Vision Powered: Uses real-time webcam input with MediaPipe hand tracking.

Multi-Module System: Includes Presentation Controller, Virtual Drawing Board, and Image Zoomer.

Web-Triggered Execution: Django frontend triggers backend computer vision logic.

Beginner-Friendly Architecture: Rule-based gesture logic instead of complex ML models.

🛠️ Tech Stack

🐍 Python: Core programming language

🎥 OpenCV: Webcam capture, image processing, drawing

✋ MediaPipe: Hand landmark detection

🧠 CVZone: Simplified gesture detection utilities

🌐 Django: Web framework (URL routing, views, templates)

🎨 HTML/CSS/Bootstrap: Frontend interface

🎉 Features

Gesture-Controlled Presentation

Next / Previous slide using finger gestures

Slide navigation without keyboard or mouse

Virtual Drawing Board

Draw in the air using index finger

Clear, write, and interact with a virtual canvas

Image Zoomer

Zoom in/out using two-hand distance gestures

Real-time scaling based on finger movement

Real-Time Webcam Processing

Fast frame processing with OpenCV

Smooth gesture recognition

Web Interface

Select presentation folders and images

Trigger modules from browser UI

🧑‍💻 Usage Guide
1️⃣ Start the Django Server
python manage.py runserver

2️⃣ Access the Web Interface

Open your browser and go to:

http://127.0.0.1:8000/

3️⃣ Use the System

📂 Select a presentation or image folder

✋ Use hand gestures to:

Change slides

Draw on the virtual board

Zoom images

❌ Press Q to exit any module

⚠️ Note:
Webcam interaction is handled by OpenCV on the local machine, not inside the browser.

📂 Folder Structure
PresentationPal/
├── manage.py
├── presentationpal/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── home/
│   ├── views.py
│   ├── urls.py
│   └── templates/
│       └── index.html
├── static/
│   ├── css/
│   └── images/
├── presentations/
├── images/
├── requirements.txt
└── README.md

🧠 How It Works (High Level)

User clicks an option on the web interface

Django routes the request to a backend view

The view triggers OpenCV + MediaPipe logic

Webcam captures frames and detects hand landmarks

Gestures are mapped to actions (slide change, draw, zoom)

OpenCV displays output in a desktop window

⚙️ Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/09Namratakhade/Presentation_Pal.git
cd PresentationPal

2️⃣ Create a Virtual Environment
python -m venv venv

3️⃣ Activate the Virtual Environment

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Run the Project
python manage.py runserver

📈 Future Scope

Browser-based webcam streaming (WebRTC)

REST API integration

Gesture customization by users

Voice + gesture hybrid controls

Cloud-enabled demo mode

Mobile-friendly interface

![](https://github.com/09Namratakhade//Presentation-Pal/HandBasedController/1.png) 
![](https://github.com/09Namratakhade//Presentation-Pal/HandBasedController/2.png) 
![](https://github.com/09Namratakhade//Presentation-Pal/HandBasedController/3.png)
![](https://github.com/09Namratakhade//Presentation-Pal/HandBasedController/4.png)

📚 References

OpenCV Documentation

MediaPipe Hands

CVZone

Django Documentation

📝 License

This project is licensed under the MIT License.

👩‍💻 Author

Created by
Namrata Khade
🔗 GitHub: 09Namratakhade

🔗 LinkedIn: (add your profile link here)

✨ Gesture-Driven Interaction Made Simple!