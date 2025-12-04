import pyrealsense2 as rs
import numpy as np
import cv2

# 1. Konfiguracija pipelinea
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 2. Pokretanje
pipeline.start(config)

try:
    while True:
        # 3. Čekaj frameove
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 4. Pretvori u numpy sliku (BGR za OpenCV)
        frame = np.asanyarray(color_frame.get_data())

        cv2.imshow("RealSense - color", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC za izlaz
            break
finally:
    # 5. Čišćenje
    pipeline.stop()
    cv2.destroyAllWindows()
