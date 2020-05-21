from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
import argparse
import os
import cv2
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import time

def classify_frame(net, inputQueue, outputQueue):
    # keep looping
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            # Get input and output tensors.
            input_details = net.get_input_details()
            output_details = net.get_output_details()
            frame = inputQueue.get()
            video_resized = cv2.resize(frame, (224, 224))
            img = np.array(video_resized)
            input_data = np.expand_dims(img, axis=0)

            # Point the data to be used for testing and run the interpreter
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Obtain results and map them to the classes
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]

            # Get indices of the top k results
            top_k_indices = np.argsort(predictions)[::-1][:top_k_results]
            #print(top_k_indices)
            #print("##")
            #print(predictions)
            outputQueue.put(predictions)
            
parser = argparse.ArgumentParser(description='Image Classification')
parser.add_argument('--model_path', type=str, help='Specify the model path', required=True)
parser.add_argument('--label_path', type=str, help='Specify the label map', required=True)
parser.add_argument('--top_k', type=int, help='How many top results', default=2)

args = parser.parse_args()

model_path = args.model_path 
label_path = args.label_path 
top_k_results = args.top_k

with open(label_path, 'r') as f:
    labels = list(map(str.strip, f.readlines()))

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()

inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(interpreter, inputQueue,
    outputQueue,))
p.daemon = True
p.start()

#IMAGE_NAME = 'miperson.mp4'
# IMAGE_NAME = 'person.mp4'
# IMAGE_NAME = 'per-nop.mp4'
IMAGE_NAME = 'nop.mp4'
#IMAGE_NAME = 'dnop.mp4'
#IMAGE_NAME = 'dperson.mp4'

CWD_PATH = os.getcwd()
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
video = cv2.VideoCapture(PATH_TO_IMAGE)
#time.sleep(2.0)
fps = FPS().start()
print(PATH_TO_IMAGE)
while(video.isOpened()):
    ret,frame = video.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == True:        
        if inputQueue.empty():
            inputQueue.put(frame)
        # if the output queue *is not* empty, grab the detections
        if not outputQueue.empty():
            detections = outputQueue.get()
        
        predicted_label = np.argmax(detections)
        #print(detections)
        #print(predicted_label)
        #for i in range(top_k_results):
          #   print(labels[top_k_indices[i]], predictions[top_k_indices[i]] / 255.0)
        if detections is not None:
            label = '%s: %d%%' % (labels[predicted_label], int((detections[predicted_label]/255.0)*100))
            cv2.putText(frame,label,(20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            frame=cv2.resize(frame,(500,500))
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) == ord('q'):
                break
            # update the FPS counter
            fps.update()
    else:
        cv2.destroyAllWindows()
        break
    
# stop the timer and display FPS information
fps.stop()
time.sleep(1)
print(p,p.is_alive())
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("process id: ",os.getpid())
print("parent process:",os.getppid())
#p.terminate()
p.kill()
#p.join()
video.release()

