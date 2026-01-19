#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Display stereo depth map from Oak D camera')
parser.add_argument('--min_z', type=int, default=500,
                    help='Minimum depth (Z) in millimeters (default: 500)')
parser.add_argument('--max_z', type=int, default=5000,
                    help='Maximum depth (Z) in millimeters (default: 5000)')
parser.add_argument('--min_x', type=float, default=0.0,
                    help='Minimum X coordinate as percentage 0-100, 0=left edge (default: 0)')
parser.add_argument('--max_x', type=float, default=100.0,
                    help='Maximum X coordinate as percentage 0-100, 100=right edge (default: 100)')
parser.add_argument('--min_y', type=float, default=0.0,
                    help='Minimum Y coordinate as percentage 0-100, 0=top edge (default: 0)')
parser.add_argument('--max_y', type=float, default=100.0,
                    help='Maximum Y coordinate as percentage 0-100, 100=bottom edge (default: 100)')
parser.add_argument('--debug', action='store_true',
                    help='Show debug info and boundary lines on screen')
args = parser.parse_args()

# Depth range in millimeters
MIN_Z = args.min_z
MAX_Z = args.max_z

# Screen space coordinates (percentage 0-100)
MIN_X_PCT = args.min_x
MAX_X_PCT = args.max_x
MIN_Y_PCT = args.min_y
MAX_Y_PCT = args.max_y

# Debug mode
DEBUG = args.debug

# Validate ranges
if MIN_Z >= MAX_Z:
    print("Error: min_z must be less than max_z")
    exit(1)
if MIN_X_PCT >= MAX_X_PCT:
    print("Error: min_x must be less than max_x")
    exit(1)
if MIN_Y_PCT >= MAX_Y_PCT:
    print("Error: min_y must be less than max_y")
    exit(1)
if not (0 <= MIN_X_PCT <= 100 and 0 <= MAX_X_PCT <= 100):
    print("Error: X values must be between 0 and 100")
    exit(1)
if not (0 <= MIN_Y_PCT <= 100 and 0 <= MAX_Y_PCT <= 100):
    print("Error: Y values must be between 0 and 100")
    exit(1)

# Create pipeline
pipeline = dai.Pipeline()

# Create RGB camera node
rgbCam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

# Create stereo camera nodes
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Create stereo depth node
stereo = pipeline.create(dai.node.StereoDepth)

# Link cameras to stereo depth node
monoLeftOut = monoLeft.requestFullResolutionOutput()
monoRightOut = monoRight.requestFullResolutionOutput()
monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

# Configure stereo depth
stereo.setRectification(True)
stereo.setExtendedDisparity(True)  # Better for close-range objects
stereo.setLeftRightCheck(True)     # Better occlusion handling
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align depth to RGB camera
stereo.setOutputSize(640, 480)  # Set depth output size (must be multiple of 16)

# Request RGB output (matching depth size)
rgbOut = rgbCam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)

# Create output queues
rgbQueue = rgbOut.createOutputQueue()
depthQueue = stereo.depth.createOutputQueue()

print(f"Depth range (Z): {MIN_Z}mm ({MIN_Z/1000}m) to {MAX_Z}mm ({MAX_Z/1000}m)")
print(f"X range: {MIN_X_PCT}% to {MAX_X_PCT}%")
print(f"Y range: {MIN_Y_PCT}% to {MAX_Y_PCT}%")
print(f"Debug mode: {'ON' if DEBUG else 'OFF'}")
print("Press 'q' to quit")

# Start pipeline
with pipeline:
    pipeline.start()
    
    while pipeline.isRunning():
        # Get RGB frame
        rgbFrame = rgbQueue.get()
        rgb = rgbFrame.getCvFrame()
        
        # Get depth frame
        depth = depthQueue.get()
        assert isinstance(depth, dai.ImgFrame)
        depthFrame = depth.getFrame()
        
        # Resize depth to match RGB if needed
        if depthFrame.shape[:2] != rgb.shape[:2]:
            depthFrame = cv2.resize(depthFrame, (rgb.shape[1], rgb.shape[0]))
        
        # Get frame dimensions
        height, width = rgb.shape[:2]
        
        # Convert percentage to pixel coordinates
        MIN_X = int(width * MIN_X_PCT / 100.0)
        MAX_X = int(width * MAX_X_PCT / 100.0)
        MIN_Y = int(height * MIN_Y_PCT / 100.0)
        MAX_Y = int(height * MAX_Y_PCT / 100.0)
        
        # Create depth mask (Z range)
        depth_mask = (depthFrame >= MIN_Z) & (depthFrame <= MAX_Z)
        
        # Create spatial mask (X, Y range)
        spatial_mask = np.zeros((height, width), dtype=bool)
        spatial_mask[MIN_Y:MAX_Y, MIN_X:MAX_X] = True
        
        # Combine masks (must be within Z range AND within X,Y bounds)
        combined_mask = depth_mask & spatial_mask
        
        # Apply mask to RGB frame (show only pixels within range)
        maskedRgb = rgb.copy()
        maskedRgb[~combined_mask] = [0, 0, 0]  # Black out pixels outside range
        
        # Debug visualization
        if DEBUG:
            # Draw boundary lines
            # Vertical lines for X boundaries
            cv2.line(maskedRgb, (MIN_X, 0), (MIN_X, height), (0, 255, 0), 2)
            cv2.line(maskedRgb, (MAX_X, 0), (MAX_X, height), (0, 255, 0), 2)
            
            # Horizontal lines for Y boundaries
            cv2.line(maskedRgb, (0, MIN_Y), (width, MIN_Y), (0, 255, 0), 2)
            cv2.line(maskedRgb, (0, MAX_Y), (width, MAX_Y), (0, 255, 0), 2)
            
            # Draw debug text in bottom right corner
            debug_text = [
                f"Z: {MIN_Z}-{MAX_Z}mm",
                f"X: {MIN_X_PCT:.1f}-{MAX_X_PCT:.1f}% ({MIN_X}-{MAX_X}px)",
                f"Y: {MIN_Y_PCT:.1f}-{MAX_Y_PCT:.1f}% ({MIN_Y}-{MAX_Y}px)",
            ]
            
            # Calculate text size and position (bottom right)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            line_height = 20
            padding = 10
            
            y_offset = height - padding
            for line in reversed(debug_text):
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                x_pos = width - text_size[0] - padding
                y_pos = y_offset
                
                # Draw background rectangle for text
                cv2.rectangle(maskedRgb, 
                            (x_pos - 5, y_pos - text_size[1] - 5),
                            (x_pos + text_size[0] + 5, y_pos + 5),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(maskedRgb, line, (x_pos, y_pos), 
                           font, font_scale, (0, 255, 0), thickness)
                
                y_offset -= line_height
        
        # Display
        cv2.imshow("Masked Video Feed", maskedRgb)
        
        # Break with 'q'
        key = cv2.waitKey(1)
        if key == ord('q'):
            pipeline.stop()
            break

cv2.destroyAllWindows()