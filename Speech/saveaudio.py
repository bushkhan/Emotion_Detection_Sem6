import cv2
import numpy as np

# capture audio from the default device
audio_capture = cv2.VideoCapture(0)

# disable the Microsoft Media Foundation backend
audio_capture.set(cv2.CAP_PROP_BACKEND, cv2.CAP_DSHOW)

# set the codec and create a VideoWriter object
codec = cv2.VideoWriter_fourcc(*'mp4a')
output_file = cv2.VideoWriter('output.mp3', codec, 44100, (1, 1), False)

while True:
    # read a frame from the audio capture
    ret, frame = audio_capture.read()

    if ret:
        # extract the audio from the frame
        audio_frame = np.array(frame[:, :, 0])

        # write the audio frame to the output file
        output_file.write(audio_frame)
    else:
        break

# release the resources
audio_capture.release()
output_file.release()
