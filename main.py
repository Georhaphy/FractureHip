import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://img5.pic.in.th/file/secure-sv1/smsk-1e26f337bb6ec6813.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)


model = YOLO('bestv8.pt')
object_names = list(model.names.values())
result = []

st.title("Sakhon Frax")
img_file = st.file_uploader("เปิดไฟล์ภาพ")

if img_file is not None:    
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    result = model.predict(img)
    
   



for detection in result[0].boxes.data:
   x0, y0 = (int(detection[0]), int(detection[1]))
   x1, y1 = (int(detection[2]), int(detection[3]))
   score = round(float(detection[4]), 2)
   cls = int(detection[5])
   object_name =  model.names[cls]
   label = f'{object_name} {score}'  
   
   if  model.names[cls] == 'Fracture' :
       cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
       cv2.putText(img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
   else :
       cv2.rectangle(img (x0, y0), (x1, y1), (0, 255, 0), 2)
       cv2.putText(img, label, (x0, y0 - 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
    
    
