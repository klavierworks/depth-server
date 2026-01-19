#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Display stereo depth map from Oak D camera')
parser.add_argument('--min_distance', type=int, default=500,
                    help='Minimum depth distance in millimeters (default: 500)')
parser.add_argument('--max_distance', type=int, default=5000,
                    help='Maximum depth distance in millimeters (default: 5000)')
args = parser.parse_args()

# Depth range in millimeters
MIN_DISTANCE = args.min_distance
MAX_DISTANCE = args.max_distance

# Validate range
if MIN_DISTANCE >= MAX_DISTANCE:
    print("Error: min_distance must be less than max_distance")
    exit(1)

# Create pipeline
pipeline = dai.Pipeline()

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

# Create output queue for depth
depthQueue = stereo.depth.createOutputQueue()

# Create colormap for visualization
colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
colorMap[0] = [0, 0, 0]  # Make zero-depth pixels black

print(f"Depth range: {MIN_DISTANCE}mm ({MIN_DISTANCE/1000}m) to {MAX_DISTANCE}mm ({MAX_DISTANCE/1000}m)")
print("Press 'q' to quit")

# Start pipeline
with pipeline:
    pipeline.start()
    
    while pipeline.isRunning():
        # Get depth frame
        depth = depthQueue.get()
        assert isinstance(depth, dai.ImgFrame)
        depthFrame = depth.getFrame()
        
        # Create mask for values within range
        mask = (depthFrame >= MIN_DISTANCE) & (depthFrame <= MAX_DISTANCE)
        
        # Normalize depth values within range to 0-255
        depthNormalized = np.zeros_like(depthFrame, dtype=np.uint8)
        depthNormalized[mask] = ((depthFrame[mask] - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE) * 255).astype(np.uint8)
        
        # Apply colormap
        colorizedDepth = cv2.applyColorMap(depthNormalized, colorMap)
        
        # Black out values outside range
        colorizedDepth[~mask] = [0, 0, 0]
        
        # Display
        cv2.imshow("Stereo Depth", colorizedDepth)
        
        # Break with 'q'
        key = cv2.waitKey(1)
        if key == ord('q'):
            pipeline.stop()
            break

cv2.destroyAllWindows()