import cv2
import numpy as np
from controller import Robot, Camera, Keyboard
from vehicle import Car, Driver
from tensorflow.keras.models import load_model
import os


# === Configuration ===
MAX_SPEED = 60.0  # km/h
MIN_SPEED = 30.0  # km/h
#STEERING_STEP = 0.01  # smooth turn increments

# === Helper Functions ===
def apply_roi(image):
    h = image.shape[0]
    return image[int(h * 0.35):int(h * 0.9), :, :]

#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# set target speed
def set_speed(kmh):
    global speed            #robot.step(50)


def main():
    # === Webots Initialization ===
    # Create the Robot instance.
    robot = Robot()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create keyboard instance
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Create camera instance
    camera = robot.getDevice('camera_center')
    camera.enable(timestep)  # ms

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    model_path = os.path.join(script_dir, 'nvidia_model.keras')

    model = load_model(model_path, compile=False, safe_mode=False)

    #initial angle and speed 
    steering_angle = 0.0
    #steering_target = 0.0
    current_speed = 0.0
    target_speed = 0.0  # To start press UP or DOWN arrow keys

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Preprocess like in image capture
        image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        cropped = apply_roi(image_bgr)
        resized = cv2.resize(cropped, (200, 66))

        # Preprocess like in training
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) #Image shape: (66, 200, 3)
        input_tensor = np.expand_dims(img, axis=0) # Input tensor shape: (1, 66, 200, 3)
        #img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Predict and control
        steering_angle = float(model.predict(input_tensor)) #Expected shape (None, 66, 200, 3)
        print(f"Predicted steering angle: {steering_angle:.4f}")
        

        # Process keyboard input
        key = keyboard.getKey()
        if key == Keyboard.UP:
            target_speed = MAX_SPEED
        elif key == Keyboard.DOWN:
            target_speed = MIN_SPEED
        elif key == ord('S'):
            print("🛑 Stop key pressed. Exiting.")
            break

        # Gradual speed update
        if current_speed < target_speed:
            current_speed = min(current_speed + 0.5, target_speed)
        elif current_speed > target_speed:
            current_speed = max(current_speed - 0.5, target_speed)

        # Smooth steering update
        # if steering_angle < steering_target:
        #     steering_angle = min(steering_angle + STEERING_STEP, steering_target)
        # elif steering_angle > steering_target:
        #     steering_angle = max(steering_angle - STEERING_STEP, steering_target)

        #update angle and speed
        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(target_speed)

if __name__ == "__main__":
    main()