import pickle
from pathlib import Path
import numpy as np

from camera import take_picture
from facenet import process_image

DB_FILE = Path(__file__).parent / "face_database.pkl"

class Profile:
    def __init__(self, name):
        self.name = name
        self.descriptors = []

    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)

    def mean_descriptor(self):
        if len(self.descriptors) == 0:
            return None

        return np.mean(self.descriptors, axis=0)

def load_database():
    if not DB_FILE.exists():
        return {}
    with open(DB_FILE, 'rb') as f:
        return pickle.load(f)

data = load_database()

def save_database():
    with open(DB_FILE, 'wb') as f:
        pickle.dump(data, f)

def get_img_vector(pic=None):
    if pic is None:
        pic = take_picture()

    _, descriptors, probabilities = process_image(pic)
    if len(descriptors) == 0:
        print("No face detected in the photo.")
        return None

    best = int(np.argmax(probabilities))
    return descriptors[best] 

def add_data(name, pic=None):
    name = name.strip()

    if not name:
        raise ValueError("Name cannot be empty.")
    
    vector = get_img_vector(pic)
    if vector is None:
        print("Could not add since no face was detected.")
        return None

    if name not in data:
        data[name] = Profile(name)
        
    data[name].add_descriptor(vector)
    save_database()
    print("Addition was successful.")

def remove_profile(name):
    name = name.strip()

    if name not in data:
        print(f"{name} is not in the database.")
        return False

    del data[name]
    save_database()

    print(f"Successfully removed {name} from the database.")
    return True