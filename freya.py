from mimetypes import init
import pandas as pd
import os
import requests
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


class DatasetDownloader():
    def __init__(self) -> None:
        self.clips = pd.read_csv('./clips.csv')
        self.dataset_path = os.path.join(os.getcwd(), 'dataset')
        self.base_url = 'https://lsfb.info.unamur.be/static/datasets/LSFB/lsfb_isol/{}'

    def download_landmark_file(self, file_url, filename, location):
        try:
            r = requests.get(file_url)
        except:
            print('Cannot dowload file: {}'.format(filename))
            return False
        open(os.path.join(location, filename), 'wb').write(r.content)
        return True

    def purge_folder(self, word):
        entry_path = os.path.join(self.dataset_path, word)
        files_to_del = [file for file in os.listdir(entry_path) if file.endswith('.mp4')]
        for file in files_to_del:
            path = os.path.join(self.dataset_path, word, file)
            os.remove(path)
        print('--> Purged {} .mp4 files in folder "{}"'.format(len(files_to_del), word))

    def download_word(self, word):
        entry_path = os.path.join(self.dataset_path, word)
        try:
            os.mkdir(entry_path)
        except:
            print('Folder already exists.')
        
        word_clips = self.clips[self.clips.gloss.isin([word])].copy()
        word_clips.gloss.unique()
        # drop uneeded colums
        relative_paths = word_clips.loc[:, :'relative_path'].drop(['hand'], axis=1)
        # Loop through entries and extract all landmarks for each video
        # constructs a dict with the entry word as key and a list of lists of paths to the landmark files
        relative_paths_dict = {}
        data = relative_paths[relative_paths.gloss == word].drop(['gloss'], axis=1)
        relative_paths_dict[word] = data.relative_path


        print('-> Acessing data for: {}'.format(word), end='\r')

        for video in tqdm(relative_paths_dict[word], desc='Videos download'):
            filename = video.split('/')[-1]
            video_url = self.base_url.format(video)
            self.download_landmark_file(video_url, filename, os.path.join(self.dataset_path, word))

class KeypointsExtractor():
    def __init__(self, videos_path) -> None:
        self.dataset_path = videos_path

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def draw_styled_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS, 
                                    self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                    self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=1))
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                    self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    
    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
        return np.concatenate([pose, face, lh, rh])

    def extract_data_for_word(self, word):
        videos_path = os.path.join(self.dataset_path, word)
        try:
            video_files = os.listdir(videos_path)
        except:
            print("Cannot access data for word '{}'. Are you sure the folder exists?".format(word))
        
        # extracted_keypoints = []
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            file_num = 0
            for file in video_files:
                os.mkdir(os.path.join(videos_path, str(file_num)))
                frame_num = 0
                # vid_data = []
                cap = cv2.VideoCapture(os.path.join(videos_path, file))
                while True:
                    ret, frame = cap.read()
                    if ret == True:
                        # Make detections
                        image, results = self.mediapipe_detection(frame, holistic)
                    
                        # Draw landmarks
                        self.draw_styled_landmarks(image, results)
                        
                        # 2. Prediction logic
                        keypoints = self.extract_keypoints(results)
                        npy_path = os.path.join(videos_path, str(file_num), str(frame_num))
                        np.save(npy_path, keypoints)
                        frame_num += 1
                        # vid_data.append(keypoints)
                        cv2.imshow("VIDEO", image)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    else:
                        break
                # print(len(vid_data))
                # extracted_keypoints.append(vid_data)
                file_num += 1
                cap.release() 
                cv2.destroyAllWindows()


if __name__=='__main__':
    d = DatasetDownloader()
    ke = KeypointsExtractor(videos_path='./dataset')
    words_to_dowload = ['AMI', 'DIRE', 'AUSSI']
    for w in words_to_dowload:
        d.download_word(w)
        ke.extract_data_for_word(w)
        d.purge_folder(w)
        