import os
import shutil
import subprocess
import time
from PIL import Image
from ultralytics import YOLO
from multiprocessing import Process, Manager

# Initialize the YOLO model for the first process
model1 = YOLO('/home/yogesh/yolov8/ultralytics/runs/detect/train18/weights/best.pt')

# Initialize the YOLO model for the second process
model2 = YOLO('/home/yogesh/yolov8/ultralytics/runs/detect/train16/weights/best.pt')  # Replace with the path to your second model's weights

# Paths for input folders and output folders for both processes
input_folder1 = '/home/yogesh/yolov8/ultralytics/watchdog/input'
input_folder2 = '/home/yogesh/yolov8/ultralytics/watchdog/input2'
source_folder = '/home/yogesh/yolov8/ultralytics/watchdog/item'  # Update this with the actual source folder path
output_folder1 = '/home/yogesh/yolov8/ultralytics/watchdog/output1'
output_folder2 = '/home/yogesh/yolov8/ultralytics/watchdog/output2'

# Create the output folders if they don't exist
os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)

# Initialize a set to keep track of processed images for both processes
processed_images1 = set()
processed_images2 = set()

def process_images_with_model1(input_folder, output_folder, results1):
    while True:
        try:
            # Run rsync command to sync the input folder with the source folder
            subprocess.run(['rsync', '-av', source_folder + '/', input_folder])

            # List all files in the input folder for model 1
            image_files = os.listdir(input_folder)

            # Filter out files that have already been processed
            new_image_files = [img for img in image_files if img not in processed_images1]

            # Process each new image with model 1
            for image_file in new_image_files:
                image_path = os.path.join(input_folder, image_file)

                # Load the image using PIL
                image = Image.open(image_path)

                # Perform object detection with model 1
                results1[0] = model1.predict(source=image, project=output_folder, name='test1', exist_ok=True, save_crop=True, save=True)

                # Add the processed image to the set for model 1
                processed_images1.add(image_file)

                # Move the processed image from input to another folder for model 1
                shutil.move(image_path, os.path.join(output_folder, image_file))

            # Sleep for a while before checking again
            time.sleep(1)  # Adjust the sleep duration as needed
        except Exception as e:
            print("Error in process_images_with_model1:", str(e))

def process_images_with_model2(input_folder, output_folder, results2):
    while True:
        try:
            # Check if the 'Target' folder exists in the output folder of model 1
            target_folder = os.path.join(output_folder, 'test1/crops/Target')
            if os.path.exists(target_folder):
                # Run rsync command to sync the 'Target' subfolder from the first model's output to input2 folder
                subprocess.run(['rsync', '-av', target_folder + '/', input_folder])

                # List all files in the input folder for model 2
                image_files = os.listdir(input_folder)

                # Filter out files that have already been processed
                new_image_files = [img for img in image_files if img not in processed_images2]

                # Process each new image with model 2
                for image_file in new_image_files:
                    image_path = os.path.join(input_folder, image_file)

                    # Load the image using PIL
                    image = Image.open(image_path)

                    # Perform object detection with model 2
                    results2[0] = model2.predict(source=image, project=output_folder, name='test2', exist_ok=True, save_crop=True, save=True)

                    # Add the processed image to the set for model 2
                    processed_images2.add(image_file)

                    # Optionally move the processed image from input to another folder for model 2
                    # shutil.move(image_path, os.path.join(output_folder, image_file))

                # Sleep for a while before checking again
                time.sleep(1)  # Adjust the sleep duration as needed
        except Exception as e:
            print("Error in process_images_with_model2:", str(e))

if __name__ == '__main__':
    with Manager() as manager:
        # Create shared lists to store results for each process
        results1 = manager.list([None])  # Initialize with None
        results2 = manager.list([None])  # Initialize with None

        # Create separate processes for each model, passing results and folder paths as arguments
        process1 = Process(target=process_images_with_model1, args=(input_folder1, output_folder1, results1,))
        process2 = Process(target=process_images_with_model2, args=(input_folder2, output_folder2, results2,))

        # Start the processes
        process1.start()
        process2.start()

        try:
            # Wait for both processes to finish (this won't happen in this script since they run indefinitely)
            process1.join()
            process2.join()
        except KeyboardInterrupt:
            process1.terminate()
            process2.terminate()
