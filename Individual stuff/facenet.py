import numpy as np
from facenet_models import FacenetModel

model = FacenetModel()


def detecting_faces(pic, threshold = 0.3):     # variable "pic" is the output of Ella's code. should be an array of some sort

    boxes, probabilities, landmarks = model.detect(pic)

    if boxes is None:
        return (np.empty((0, 4)), np.array([]), np.empty((0, 5, 2)))

    confident_faces = probabilities >= threshold
    
    return (boxes[confident_faces], probabilities[confident_faces], landmarks[confident_faces])


def describing_faces(pic, boxes):

    if len(boxes) == 0:
        return np.empty((0, 512))
    
    return model.compute_descriptors(pic, boxes)
 
def process_image(pic):
    
    boxes, probabilities, _ = detecting_faces(pic)
    descriptors = describing_faces(pic, boxes)

    return boxes, descriptors, probabilities


# testing with a photo of a group of 3 

# from skimage.io import imread

# pic = imread("/Users/newton/Desktop/test.jpg")
# boxes, descriptors, probabilities = process_image(pic)

# print(f"Detected {len(boxes)} faces.")
# print("Boxes shape:", boxes.shape)
# print("Descriptors shape:", descriptors.shape)
# print("Probabilities:", probabilities)
    