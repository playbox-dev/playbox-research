import argparse
import cv2
import time
from pbr.camera import Camera
from pbr.stitcher import VideoStitcher

def main(camera1_src, camera2_src, output_file, recording_time=10):
    camera1 = Camera(src=camera1_src)
    camera2 = Camera(src=camera2_src)

    # Setup codec and VideoWriter for recording
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    camera1_out = cv2.VideoWriter('camera1_output.mp4', fourcc, 15.0, (1920, 1080))
    camera2_out = cv2.VideoWriter('camera2_output.mp4', fourcc, 15.0, (1920, 1080))

    # Record from both cameras
    start_time = time.time()
    while time.time() - start_time < recording_time:
        frame1 = camera1.get_frame()
        frame2 = camera2.get_frame()

        if frame1 is not None:
            camera1_out.write(frame1)
        if frame2 is not None:
            camera2_out.write(frame2)

    camera1_out.release()
    camera2_out.release()

    stitcher = VideoStitcher(detector="sift", confidence_threshold=0.2)

    init_writer = False

    camera1_recording = cv2.VideoCapture('camera1_output.mp4')
    camera2_recording = cv2.VideoCapture('camera2_output.mp4')

    while True:
        ret1, frame1 = camera1_recording.read()
        ret2, frame2 = camera2_recording.read()
        
        if ret1 and ret2:
            stitched = stitcher.stitch([frame1, frame2])

            if init_writer is False:
                # Initialize VideoWriter for the stitched output
                out_size = (stitched.shape[1], stitched.shape[0])
                out = cv2.VideoWriter(output_file, fourcc, 15.0, out_size)
                init_writer = True

            cv2.imwrite('stitched_frame.jpg', stitched)

            # Convert to correct depth if necessary and write to output
            stitched_correct_depth = cv2.convertScaleAbs(stitched)
            out.write(stitched_correct_depth)
        else:
            print("Finished")
            break

    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time video stitching.')
    parser.add_argument('--camera1', type=int, default=0, help='Source index for camera 1')
    parser.add_argument('--camera2', type=int, default=1, help='Source index for camera 2')
    parser.add_argument('--output', type=str, default='stitching_output.mp4', help='Output file name')
    args = parser.parse_args()
    
    main(args.camera1, args.camera2, args.output)