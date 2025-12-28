#This is the entry point of GazeGuard.
import cv2
import sys
import config  #import the config settings

def main():
    print("Initializing GazeGuard...")

    # 1. Setup Camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    print("Camera active! Press 'q' to quit.")

    # 2. Main Loop (Runs every frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip the frame horizontally (like a mirror)
        frame = cv2.flip(frame, 1)

        # Show the video feed
        cv2.imshow("GazeGuard - Camera Test", frame)

        # Exit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 3. Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()