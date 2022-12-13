"""
align.py

Align camera by live-streaming AR Tag locations (one can use this script to either save the positions of the tags
relative to the current camera vantage for later realignment, or for realigning the camera).
"""
import json
import time
from pathlib import Path
from typing import Tuple

import cv2
from tap import Tap


# Helpful Mappings & Constants
GROUND_TRUTH_CENTERS = Path("demos/centers.json")

# noinspection PyUnresolvedReferences
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArgumentParser(Tap):
    # fmt: off
    mode: str = "align"                             # Mode for aligning -- in < detect | align >
    n_tags: int = 2                                 # Minimum number of tags to use for detection/alignment
    marker_ids: Tuple[int, int] = (0, 1)            # Actual Marker IDs of the tags to use in detection

    # AruCo Specific Parameters
    aruco_type: str = "DICT_4X4_50"                 # Type of AruCo tag (defined in cv2) to detect/align against
    # fmt: on


def align() -> None:
    # Parse arguments...
    args = ArgumentParser().parse_args()

    # Load the ArUCo Dictionary and grab the corresponding parameters for the specified AruCo tag type...
    print(f"[*] Attempting to Detect AruCo Tag Type {args.aruco_type}...")
    ar_dict, ar_params = cv2.aruco.Dictionary_get(ARUCO_DICT[args.aruco_type]), cv2.aruco.DetectorParameters_create()

    # Load the ground-truth centers if possible...
    ground_truth = None
    if GROUND_TRUTH_CENTERS.exists():
        with open(GROUND_TRUTH_CENTERS) as f:
            ground_truth = json.load(f)

        # Run sanity checks...
        print("[*] Verifying ground truth centers are valid...")
        assert len(ground_truth) == args.n_tags
        for k in args.marker_ids:
            assert str(k) in ground_truth

    # Start VideoStream (and purge buffer for the first three seconds...)
    print("[*] Starting Video Stream...")
    vs, start_time = cv2.VideoCapture(0), time.time()

    # Burn-In (cv2.imshow is weird...)
    for _ in range(7):
        _, frame = vs.read()
        cv2.imshow("Frame", frame)

    # Loop over frames in Video Stream...
    while True:
        # Grab the frame and resize to a maximum width of 1000 pixels...
        _, frame = vs.read()
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AruCo Markers in the Input Frame
        corners, ids, rejected = cv2.aruco.detectMarkers(grayscale, ar_dict, parameters=ar_params)
        detected_centers = {}

        # Verify that at least one marker was detected... if not, show frame, and reload the loop!
        if len(corners) == 0:
            print("[*] No AruCo Tag Detected... check frame for obstructions...")

        # Otherwise, let's visualize the AruCo Markers...
        else:
            if args.mode == "detect":
                print("[*] Tags Detected - press `s` (while cv2 frame in focus) to save them as ground-truth centers!")

            # Flatten the AruCo IDs list to help with mapping to positions...
            ids = ids.flatten().tolist()

            # Loop over the detected AruCo corners...
            for marker_corner, marker_id in zip(corners, ids):
                # Extract the marker corners :: always returned clockwise from top-left --> bottom-left
                top_left, top_right, bottom_right, bottom_left = marker_corner.reshape((4, 2)).astype(int)

                # Draw the bounding box of each AruCo detection...
                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                # Compute & draw the center (x, y) coordinated of each marker...
                center = ((top_left + bottom_right) / 2.0).astype(int)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)

                # Draw the ArUco Marker ID directly on the frame...
                cv2.putText(
                    frame, str(marker_id), (top_left[0], top_left[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                # Update the set of detected centers...
                detected_centers[marker_id] = center.tolist()

            # If in "alignment" mode, drop into an interactive loop showing deltas per marker!
            if args.mode == "align":
                # Only flush every 2 seconds...
                if time.time() - start_time >= 2:
                    for m_id in args.marker_ids:
                        if m_id in detected_centers:
                            detected_center, true_center = detected_centers[m_id], ground_truth[str(m_id)]

                            # Compute deltas per direction (X/Y) per marker...
                            dx, dy = true_center[0] - detected_center[0], true_center[1] - detected_center[1]
                            x_adjust, y_adjust = "left" if dx > 0 else "right", "up" if dy > 0 else "down"
                            print(f"Marker {m_id}: dx = {abs(dx)} {x_adjust} || dy = {abs(dy)} {y_adjust}")
                    print("\n")
                    start_time = time.time()

        # Show the output frame...
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Assert that the detected centers are valid...
        if args.mode == "detect" and len(detected_centers) >= args.n_tags:
            detected_marker_ids, valid = set(detected_centers.keys()), True
            for k in args.marker_ids:
                # If a desired marker_id is not in the set, we're invalid...
                valid &= k in detected_marker_ids

            # If not valid... continue
            if not valid:
                continue

        # If `s` is pressed --> we have a set of markers, log them to file and break...
        if key == ord("s") and args.mode == "detect":
            print("[*] Saving Detected Centers...")
            GROUND_TRUTH_CENTERS.parent.mkdir(parents=True, exist_ok=True)
            with open(GROUND_TRUTH_CENTERS, "w") as f:
                json.dump(detected_centers, f, indent=4)
            break

        # If `q` is pressed, break from the loop!
        elif key == ord("q"):
            break

    # Cleanup & Exit...
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    align()
