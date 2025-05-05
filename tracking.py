import streamlit as st
import cv2 as cv
import tempfile
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

f = st.file_uploader("Загрузите видео")


if f:
    st.write("Ваше видео")
    st.video(f)

    progress_text = "Идёт обработка"
    my_bar = st.progress(0, text=progress_text)

    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())

    cap = cv.VideoCapture(tfile.name)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    number_of_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

    stframe = st.empty()

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Video ended")
            break
        
        results = model.track(frame, persist=True, save_txt=True)
                        
        counter += 1
        my_bar.progress(counter / number_of_frames)

        frame_ = results[0].plot()
        stframe.image(frame_)
        out.write(frame_)

    out.release()
    
    with open("output.mp4", "rb") as f:
        st.download_button("Скачать видео с трекингом", f, file_name="output.mp4")