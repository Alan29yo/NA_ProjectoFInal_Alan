from controller import Robot, Camera, Keyboard
from vehicle import Driver
import os
import csv
import cv2
import numpy as np
from datetime import datetime

# === Configuration ===
RESIZED_SHAPE = (200, 66)   # TensorFlow input shape
TOTAL_IMAGES = 5500
ANGLE_CORRECTION = 0.15
MAX_SPEED = 30.0  # km/h
STEERING_STEP = 0.01  # smooth turn increments
IMAGE_DIR = "images"
CSV_FILE = "driving_log.csv"

# Calculate steps per capture based on FPS and timestep
robot = Robot()
driver = Driver()
keyboard = Keyboard()
timestep = int(robot.getBasicTimeStep())  # 10 ms
keyboard.enable(timestep)


# === Cameras ===
cameras = {
    "center": robot.getDevice("camera_center"),
    "left": robot.getDevice("camera_left"),
    "right": robot.getDevice("camera_right")
}
for cam in cameras.values():
    cam.enable(timestep)

cam_height = cameras["center"].getHeight()  # Cameras have the same height and width
cam_width = cameras["center"].getWidth()

# === Output Setup ===
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
image_dir = os.path.join(script_dir, IMAGE_DIR)
csv_file = os.path.join(script_dir, CSV_FILE)

os.makedirs(image_dir, exist_ok=True)

if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "steering_angle", "speed", "timestamp", "camera_position"])

# === Helper Functions ===
def apply_roi(image):
    h = image.shape[0]
    return image[int(h * 0.35):int(h * 0.9), :, :]

def save_frame(image, angle, speed, tag):
    global image_count
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{tag}_{timestamp}.png"
    filepath = os.path.join(image_dir, filename)
    rel_filepath = os.path.join(IMAGE_DIR, filename)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    cropped = apply_roi(image_bgr)
    resized = cv2.resize(cropped, RESIZED_SHAPE)
    cv2.imwrite(filepath, resized)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([rel_filepath, round(angle, 4), round(speed, 2), timestamp, tag])

    return 1

# === Main Variables ===
steering_angle = 0.0
steering_target = 0.0
current_speed = 0.0
target_speed = 0.0
image_count = 0
step_count = 0
capture_fps = 5 # Frames per second for capturing images
running = True

print(f"🚘 Speed will gradually reach {MAX_SPEED} km/h")
print("🎮 Arrows to steer smoothly | Press 'S' to stop")

# === Main Loop ===
while robot.step() != -1 and image_count < TOTAL_IMAGES and running:
    step_count += 1
    key = keyboard.getKey()

    if key == Keyboard.LEFT:
        steering_target = max(steering_target - STEERING_STEP, -0.5)
    elif key == Keyboard.RIGHT:
        steering_target = min(steering_target + STEERING_STEP, 0.5)
    elif key == Keyboard.UP:
        target_speed = MAX_SPEED
    elif key == Keyboard.DOWN:
        # 🚗 Reset steering to 0 when Down arrow is pressed
        steering_target = 0.0
        steering_angle = 0.0
    elif key in [ord(str(d)) for d in range(1, 10)]:
        # Set capture_fps to the digit pressed (1-9 FPS)
        capture_fps = int(chr(key))
    elif key == ord('S'):
        print("🛑 Stop key pressed. Exiting.")
        break

    # Gradual speed and steering update
    if current_speed < target_speed:
        current_speed = min(current_speed + 0.5, target_speed)
    elif current_speed > target_speed:
        current_speed = max(current_speed - 0.5, target_speed)

    # Smooth steering update
    if steering_angle < steering_target:
        steering_angle = min(steering_angle + STEERING_STEP, steering_target)
    elif steering_angle > steering_target:
        steering_angle = max(steering_angle - STEERING_STEP, steering_target)

    # Apply to car
    driver.setCruisingSpeed(current_speed)
    driver.setSteeringAngle(steering_angle)


    steps_per_capture = int(1000 / (capture_fps * timestep)) # 1000 ms / (FPS * timestep in ms)
    # === Image Capture ===
    if current_speed > 0.0 and step_count % steps_per_capture == 0:
        #Center, left, and right camera images
        raw_img_center = cameras["center"].getImage()
        raw_img_left = cameras["left"].getImage()
        raw_img_right = cameras["right"].getImage()
        image_count += save_frame(
            np.frombuffer(raw_img_center, np.uint8).reshape((cam_height, cam_width, 4)),
            steering_angle, current_speed, "center"
        )
        image_count += save_frame(
            np.frombuffer(raw_img_left, np.uint8).reshape((cam_height, cam_width, 4)),
            steering_angle + ANGLE_CORRECTION, current_speed, "left"
        )
        image_count += save_frame(
            np.frombuffer(raw_img_right, np.uint8).reshape((cam_height, cam_width, 4)),
            steering_angle - ANGLE_CORRECTION, current_speed, "right"
        )

        print(f"[{image_count}/{TOTAL_IMAGES}] Captured | Speed: {current_speed:.1f} km/h | Steering: {steering_angle:.2f} | Capture rate: {capture_fps} FPS")

print("✅ Data collection complete.")