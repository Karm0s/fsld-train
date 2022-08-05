import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tqdm import tqdm

from pretty_confusion_matrix import pp_matrix_from_data

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from train.utils import MediapipeUtils

class FSLModel():
    def __init__(self) -> None:
        # Create necessary attributes
        self.dataset_path = os.path.join(os.getcwd(), 'dataset')
        self.words = self.get_dataset_words()
        print('-> Creating label map.')
        self.label_map = self.create_labels(self.words)
        self.sequence_length = 30

        print('-> Creating LSTM model and setting up Tensoflow logs.')
        self.model = self.create_lstm_model()

        # Load data and setup labels
        print('-> Loading the dataset.')
        self.data, self.labels = self.load_data()

        self.X = np.array(self.data)

        print('-> Split data to training and testing sets.')
        self.y = to_categorical(self.labels).astype(int)
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.05)

        print('-> Setup Tensorflow logs.')
        self.log_dir, self.tb_callback = self.setup_tf_logs()


    def create_labels(self, words):
        label_map = {label: num for num, label in enumerate(words)}
        return label_map
    
    def get_words_list(self):
        return self.words

    def get_dataset_words(self):
        return os.listdir(self.dataset_path)
    
    def create_lstm_model(self):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.sequence_length, 1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(self.words), activation='softmax'))

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        return model

    def setup_tf_logs(self):
        log_dir = os.path.join(os.getcwd(), 'Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        return log_dir, tb_callback

    def train(self, epochs=100):
        print('-> Training model.')
        self.train_model(self.model, self.X_train, self.y_train, self.tb_callback, epochs=epochs)
        print('-> Saving model weights.')
        self.weights_path = './new_weights.h5' 
        self.save_weights(self.model, self.weights_path)

    def test_model(self):
        preds = self.model.predict(self.X_test)
        y_true = np.argmax(self.y_test, axis=1).tolist()
        print(y_true)
        y_pred = np.argmax(preds, axis=1).tolist()
        # self.confusion_matrix = multilabel_confusion_matrix(y_true, y_pred, labels=self.labels)

        pp_matrix_from_data(y_true, y_pred)

    def predict(self, data):
        return self.model.predict(data)

    def save_weights(self, model, file_path):
        model.save(file_path)

    def load_weights(self, file_path):
        self.model.load_weights(file_path)

    def train_model(self, model, X_train, y_train, callbacks, epochs= 200):
        model.fit(X_train, y_train, epochs=epochs, callbacks=callbacks)

    def sequence_padding(self, sequence, length=30):
        for frame in range(length - len(sequence)):
            sequence.append(np.zeros(1662))

    def load_data(self):
        sequences, labels = [], []

        for action in self.words:
            print('\t->Loading data for word {}'.format(action))
            num_sequences = os.listdir(os.path.join(self.dataset_path, action))
            for sequence in tqdm(range(len(num_sequences)), desc="Loading video data"):
                window = []
                sequence_length = os.listdir(os.path.join(self.dataset_path, action, str(sequence)))
                for frame_num in range(len(sequence_length)):
                    if frame_num == self.sequence_length:
                        break
                    data = np.load(os.path.join(self.dataset_path, action, str(sequence), '{}.npy'.format(frame_num)))
                    window.append(data)
                if len(window) < self.sequence_length:
                    self.sequence_padding(window, self.sequence_length)
                sequences.append(window)
                labels.append(self.label_map[action])
        
        return sequences, labels

class Main():
    def __init__(self, model_weights='') -> None:
        self.model = FSLModel()
        self.actions = self.model.get_words_list()

        if model_weights:
            self.model.load_weights(model_weights)
        else:
            self.model.train(epochs=50)
        
        self.model.test_model()

        self.mediapipe_utils = MediapipeUtils()

        self.sequence = []
        self.sentence = []
        self.detection_threshold = 0.4

    def prob_viz(self, res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        return output_frame

    def run(self):
        sentence = []
        sequence = []
        colors = [(245,117,16), (117,245,16), (16,117,245), (128,0,128), (0, 0, 0), (0,255,255)]
        capture = cv2.VideoCapture(0)
        with self.mediapipe_utils.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while capture.isOpened():

                # Read feed
                ret, frame = capture.read()

                # Make detections
                image, results = self.mediapipe_utils.mediapipe_detection(frame, holistic)
            
                # Draw landmarks
                self.mediapipe_utils.draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = self.mediapipe_utils.extract_keypoints(results)
        #         sequence.insert(0,keypoints)
        #         sequence = sequence[:30]
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                    
                    
                #3. Viz logic
                    if res[np.argmax(res)] > self.detection_threshold: 
                        if len(sentence) > 0: 
                            if self.actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(self.actions[np.argmax(res)])
                        else:
                            sentence.append(self.actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = self.prob_viz(res, self.actions, image, colors)
                    
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            capture.release()
            cv2.destroyAllWindows() 
    

if __name__=='__main__':
    app = Main(model_weights='new_weights.h5').run()
