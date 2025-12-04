import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# ---------- MediaPipe Pose ----------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ---------- RealSense konfiguracija ----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      enable_segmentation=False) as pose:

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # ----- RGB slika iz RealSense-a -----
            color_image = np.asanyarray(color_frame.get_data())  # BGR

            # MediaPipe radi u RGB
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            h, w = color_image.shape[:2]

            if results.pose_landmarks:
                # Nacrtaj skeleton
                mp_drawing.draw_landmarks(
                    color_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

                # Primjer: uzmi NOS kao keypoint
                lm = results.pose_landmarks.landmark
                nose = lm[mp_pose.PoseLandmark.NOSE.value]

                # 2D piksel koordinate na slici
                x_px = int(nose.x * w)
                y_px = int(nose.y * h)

                # Dubina u metrima iz depth frame-a
                depth_m = depth_frame.get_distance(x_px, y_px)

                # 3D koordinate u koordinatnom sustavu kamere (m)
                intr = depth_frame.profile.as_video_stream_profile().intrinsics
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [x_px, y_px], depth_m)

                # Oznaci točku na slici
                cv2.circle(color_image, (x_px, y_px), 5, (0, 0, 255), -1)

                # Ispis informacija
                info = f"Nose: ({X:.2f}, {Y:.2f}, {Z:.2f}) m"
                print(info)

            cv2.imshow("RealSense + MediaPipe Pose", color_image)

            if cv2.waitKey(1) == 27:  # ESC
                break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
