import os
import cv2
import numpy as np

def generate_sample_assets(output_dir: str = "sample_media"):
    os.makedirs(output_dir, exist_ok=True)

    bus_img_path = os.path.join(output_dir, "bus.jpg")
    img_path = os.path.join(output_dir, "traffic_sample.jpg")
    vid_path = os.path.join(output_dir, "traffic_video.mp4")

    if not os.path.exists(bus_img_path):
        print(f"Warning: {bus_img_path} not found. Please provide base image.")
        return

    base_img = cv2.imread(bus_img_path)
    bh, bw = base_img.shape[:2]

    # Crop the real bus and person from bus.jpg
    bus_crop = cv2.resize(base_img[220:730, 20:800], (320, 240))
    person_crop = cv2.resize(base_img[380:800, 30:250], (80, 180))

    width, height = 1280, 720
    fps = 20
    num_frames = 25

    # 1. Generate Static Sample Image
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (45, 48, 55) # Asphalt road
    # Road markings
    cv2.line(canvas, (640, 0), (640, height), (0, 215, 255), 4) # Yellow center
    cv2.line(canvas, (320, 0), (320, height), (220, 220, 220), 2)
    cv2.line(canvas, (960, 0), (960, height), (220, 220, 220), 2)

    # Place real bus in lane 1
    canvas[250:250+240, 120:120+320] = bus_crop
    # Place real persons
    canvas[280:280+180, 500:500+80] = person_crop
    canvas[280:280+180, 560:560+80] = person_crop

    cv2.imwrite(img_path, canvas)
    print(f"Generated realistic sample image: {img_path}")

    # 2. Generate Realistic Video with Wrong-Way Bus
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))

    for f_idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (45, 48, 55)

        # Lines
        cv2.line(frame, (640, 0), (640, height), (0, 215, 255), 4)
        cv2.line(frame, (320, 0), (320, height), (220, 220, 220), 2)
        cv2.line(frame, (960, 0), (960, height), (220, 220, 220), 2)

        # Legal vehicle moving Southbound (downwards) in Lane 1
        legal_y = int(50 + f_idx * 16)
        if legal_y + 180 < height:
            frame[legal_y:legal_y+180, 160:160+80] = person_crop

        # Wrong-Way Bus moving Northbound (UPWARDS against Southbound traffic!) in Lane 3
        bus_y = int(450 - f_idx * 14)
        if bus_y >= 0 and bus_y + 240 <= height:
            frame[bus_y:bus_y+240, 750:750+320] = bus_crop

        writer.write(frame)

    writer.release()
    print(f"Generated realistic sample video with opposing vehicle: {vid_path}")

if __name__ == "__main__":
    generate_sample_assets()
