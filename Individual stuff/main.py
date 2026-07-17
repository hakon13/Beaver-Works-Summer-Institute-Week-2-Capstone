import ssl
ssl._create_default_https_context = ssl._create_unverified_context 

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from database import (load_database, save_database, add_data, data, Profile)
from whispers import (whispers)
from image_to_rgb import (image_to_rgb_array)
from facenet import (process_image)
from sklearn.metrics.pairwise import cosine_distances

IMAGE_DIR = Path(__file__).parent / "database_images"


def display_labeled_image(image, boxes, names):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    
    for box, name in zip(boxes, names):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        rect = patches.Rectangle((x_min, y_min), width, height, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min + 5, y_max - 5, name, color='white', fontsize=12, 
                bbox=dict(facecolor='red', alpha=0.5, pad=1))
        
    plt.axis('off')
    plt.show()


def show_unknown_face_popup(image, box):
    x_min, y_min, x_max, y_max = map(int, box)
    
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)
    
    cropped_face = image[y_min:y_max, x_min:x_max]
    
    plt.figure(figsize=(3, 3))
    plt.imshow(cropped_face)
    plt.axis('off')
    plt.title("Who is this?")
    plt.show(block=False)
    plt.pause(0.1)


def match_descriptor_to_database(descriptor, db, threshold=0.5):
    best_name = "Unknown"
    min_dist = float('inf')
    
    descriptor_reshaped = descriptor.reshape(1, -1)
    
    for name, profile in db.items():
        if len(profile.descriptors) == 0:
            continue
            
        avg_vector = profile.mean_descriptor().reshape(1, -1)
        dist = cosine_distances(descriptor_reshaped, avg_vector)[0, 0]
        
        if dist < min_dist:
            min_dist = dist
            if dist < threshold:
                best_name = name
                
    return best_name


def main():
    for person_folder in IMAGE_DIR.iterdir(): 
        if not person_folder.is_dir():
            continue

        name = person_folder.name
        print("Processing " + name)
        image_count = 0

        for image_file in person_folder.iterdir():
            try:
                image = image_to_rgb_array(image_file)
            except Exception as error:
                print("Could not read " + image_file.name + ", skipping")
                print(error)
                continue
            
            print("Adding " + image_file.name)
            add_data(name, pic=image)
            image_count += 1
        
        print(f"Finished adding {name}")
        print(f"{image_count} images added")
        
    save_database()
    print("Database updated.\n")

    test_image_path = Path(__file__).parent / "test.png"
    
    if not test_image_path.exists():
        print(f"Test image '{test_image_path}' not found. Skipping inference.")
        return

    print(f"Running inference on '{test_image_path.name}'...")
    try:
        test_img_array = image_to_rgb_array(test_image_path)
    except Exception as e:
        print(f"Failed to load test image: {e}")
        return

    boxes, descriptors, probabilities = process_image(test_img_array)
    
    if len(boxes) == 0:
        print("No faces detected.")
        return

    names = []
    
    for box, descriptor in zip(boxes, descriptors):
        name = match_descriptor_to_database(descriptor, data, threshold=0.5)
        
        if name == "Unknown":
            show_unknown_face_popup(test_img_array, box)
            
            user_input = input("Unknown face. Enter a name: ").strip()
            plt.close()
            
            if user_input:
                name = user_input
                if name not in data:
                    data[name] = Profile(name)
                data[name].add_descriptor(descriptor)
                save_database()
                print(f"Saved {name} to the database!")
                
        names.append(name)

    display_labeled_image(test_img_array, boxes, names)


if __name__ == "__main__":
    main()



