#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import pyvirtualcam

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def parse_arguments():
    """Parse and validate command line arguments."""
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
    parser.add_argument('--confidence', type=int, default=200,
                        help='Confidence threshold 0-255, higher = stricter filtering (default: 200)')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug info and boundary lines on screen')
    parser.add_argument('--virtual-cam', action='store_true',
                        help='Output to virtual webcam (requires pyvirtualcam)')
    
    args = parser.parse_args()
    validate_arguments(args)
    return args


def validate_arguments(args):
    """Validate argument ranges and relationships."""
    if args.min_z >= args.max_z:
        print("Error: min_z must be less than max_z")
        exit(1)
    if args.min_x >= args.max_x:
        print("Error: min_x must be less than max_x")
        exit(1)
    if args.min_y >= args.max_y:
        print("Error: min_y must be less than max_y")
        exit(1)
    if not (0 <= args.min_x <= 100 and 0 <= args.max_x <= 100):
        print("Error: X values must be between 0 and 100")
        exit(1)
    if not (0 <= args.min_y <= 100 and 0 <= args.max_y <= 100):
        print("Error: Y values must be between 0 and 100")
        exit(1)
    if not (0 <= args.confidence <= 255):
        print("Error: Confidence threshold must be between 0 and 255")
        exit(1)


def create_pipeline():
    """Create and configure the DepthAI pipeline."""
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
    stereo.setOutputSize(CAMERA_WIDTH, CAMERA_HEIGHT)  # Set depth output size (must be multiple of 16)
    stereo.setSubpixel(True)  # Better depth precision

    # Request RGB output (matching depth size)
    rgbOut = rgbCam.requestOutput((CAMERA_WIDTH, CAMERA_HEIGHT), type=dai.ImgFrame.Type.BGR888p)

    # Create output queues
    rgbQueue = rgbOut.createOutputQueue()
    depthQueue = stereo.depth.createOutputQueue()
    confidenceQueue = stereo.confidenceMap.createOutputQueue()

    return pipeline, rgbQueue, depthQueue, confidenceQueue


def print_configuration(args):
    """Print the current configuration."""
    print(f"Depth range (Z): {args.min_z}mm ({args.min_z/1000}m) to {args.max_z}mm ({args.max_z/1000}m)")
    print(f"X range: {args.min_x}% to {args.max_x}%")
    print(f"Y range: {args.min_y}% to {args.max_y}%")
    print(f"Confidence threshold: {args.confidence}/255")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"Virtual webcam: {'ON' if args.virtual_cam else 'OFF'}")
    print("Press 'q' to quit")


def create_colormap():
    """Create colormap for depth visualization."""
    colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    colorMap[0] = [0, 0, 0]  # Make zero-depth pixels black
    return colorMap


def get_pixel_bounds(width, height, min_x_pct, max_x_pct, min_y_pct, max_y_pct):
    """Convert percentage coordinates to pixel coordinates."""
    min_x = int(width * min_x_pct / 100.0)
    max_x = int(width * max_x_pct / 100.0)
    min_y = int(height * min_y_pct / 100.0)
    max_y = int(height * max_y_pct / 100.0)
    return min_x, max_x, min_y, max_y


def create_masks(depth_frame, confidence_map, height, width, args):
    """Create depth, confidence, and spatial masks."""
    # Convert percentage to pixel coordinates
    min_x, max_x, min_y, max_y = get_pixel_bounds(
        width, height, args.min_x, args.max_x, args.min_y, args.max_y
    )
    
    # Create depth mask (Z range)
    depth_mask = (depth_frame >= args.min_z) & (depth_frame <= args.max_z)
    
    # Create confidence mask
    confidence_mask = confidence_map > args.confidence
    
    # Create spatial mask (X, Y range)
    spatial_mask = np.zeros((height, width), dtype=bool)
    spatial_mask[min_y:max_y, min_x:max_x] = True
    
    # Combine masks
    combined_mask = depth_mask & spatial_mask & confidence_mask
    
    return combined_mask, depth_mask, spatial_mask, confidence_mask, (min_x, max_x, min_y, max_y)


def apply_debug_visualization(masked_rgb, depth_frame, spatial_mask, depth_mask, 
                              confidence_mask, args, max_z, pixel_bounds, height, width):
    """Apply debug visualization including colorized depth and boundary lines."""
    min_x, max_x, min_y, max_y = pixel_bounds
    
    # Show depth colormap for pixels in X/Y range but outside Z range or low confidence
    out_of_z_range = spatial_mask & ~depth_mask & confidence_mask
    low_confidence = spatial_mask & ~confidence_mask
    
    # Normalize depth values for visualization (show up to 2x max range)
    color_map = create_colormap()
    depth_clipped = np.clip(depth_frame, 0, max_z * 2)
    depth_normalized = (depth_clipped / (max_z * 2) * 255).astype(np.uint8)
    colorized_depth = cv2.applyColorMap(depth_normalized, color_map)
    
    # Apply depth visualization to out-of-range pixels
    masked_rgb[out_of_z_range] = colorized_depth[out_of_z_range]
    masked_rgb[low_confidence] = [128, 0, 128]  # Purple for low confidence
    masked_rgb[~spatial_mask] = [0, 0, 0]  # Black for outside X/Y bounds
    
    # Draw boundary lines
    cv2.line(masked_rgb, (min_x, 0), (min_x, height), (0, 255, 0), 2)
    cv2.line(masked_rgb, (max_x, 0), (max_x, height), (0, 255, 0), 2)
    cv2.line(masked_rgb, (0, min_y), (width, min_y), (0, 255, 0), 2)
    cv2.line(masked_rgb, (0, max_y), (width, max_y), (0, 255, 0), 2)


def draw_debug_text(masked_rgb, confidence_map, spatial_mask, args, pixel_bounds, height, width):
    """Draw debug text overlay in bottom right corner."""
    min_x, max_x, min_y, max_y = pixel_bounds
    
    # Calculate confidence stats
    conf_mean = np.mean(confidence_map[spatial_mask])
    
    # Prepare debug text
    debug_text = [
        f"Z: {args.min_z}-{args.max_z}mm",
        f"X: {args.min_x:.1f}-{args.max_x:.1f}% ({min_x}-{max_x}px)",
        f"Y: {args.min_y:.1f}-{args.max_y:.1f}% ({min_y}-{max_y}px)",
        f"Conf: {args.confidence}/255 (avg: {conf_mean:.0f})",
    ]
    
    # Text rendering settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    padding = 10
    
    # Draw text from bottom to top
    y_offset = height - padding
    for line in reversed(debug_text):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x_pos = width - text_size[0] - padding
        y_pos = y_offset
        
        # Draw background rectangle
        cv2.rectangle(masked_rgb, 
                     (x_pos - 5, y_pos - text_size[1] - 5),
                     (x_pos + text_size[0] + 5, y_pos + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(masked_rgb, line, (x_pos, y_pos), 
                   font, font_scale, (0, 255, 0), thickness)
        
        y_offset -= line_height


def process_frame(rgb_queue, depth_queue, confidence_queue, args):
    """Process a single frame from all queues."""
    # Get frames
    rgb_frame = rgb_queue.get()
    rgb = rgb_frame.getCvFrame()
    
    depth = depth_queue.get()
    assert isinstance(depth, dai.ImgFrame)
    depth_frame = depth.getFrame()
    
    confidence = confidence_queue.get()
    confidence_map = confidence.getFrame()
    
    # Resize depth to match RGB if needed
    if depth_frame.shape[:2] != rgb.shape[:2]:
        depth_frame = cv2.resize(depth_frame, (rgb.shape[1], rgb.shape[0]))
    
    # Get frame dimensions
    height, width = rgb.shape[:2]
    
    # Create masks
    combined_mask, depth_mask, spatial_mask, confidence_mask, pixel_bounds = create_masks(
        depth_frame, confidence_map, height, width, args
    )
    
    # Apply mask to RGB frame
    masked_rgb = rgb.copy()
    
    if args.debug:
        apply_debug_visualization(masked_rgb, depth_frame, spatial_mask, depth_mask,
                                 confidence_mask, args, args.max_z, pixel_bounds, height, width)
        draw_debug_text(masked_rgb, confidence_map, spatial_mask, args, pixel_bounds, height, width)
    else:
        # In normal mode: hide everything outside the combined mask
        masked_rgb[~combined_mask] = [0, 0, 0]
    
    return masked_rgb


def run_processing_loop(pipeline, rgb_queue, depth_queue, confidence_queue, args, virtual_cam=None):
    """Main processing loop for camera frames."""
    pipeline.start()
    
    try:
        while pipeline.isRunning():
            masked_rgb = process_frame(rgb_queue, depth_queue, confidence_queue, args)
            
            # Send to virtual camera if enabled
            if virtual_cam:
                virtual_cam.send(cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2RGB))
            
            # Display
            cv2.imshow("Masked Video Feed", masked_rgb)
            
            # Break with 'q'
            key = cv2.waitKey(1)
            if key == ord('q'):
                pipeline.stop()
                break
    finally:
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    args = parse_arguments()
    print_configuration(args)
    
    pipeline, rgb_queue, depth_queue, confidence_queue = create_pipeline()
    
    if args.virtual_cam:
        with pyvirtualcam.Camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=60) as cam:
            print(f"Virtual webcam started: {cam.device}")
            with pipeline:
                run_processing_loop(pipeline, rgb_queue, depth_queue, confidence_queue, args, cam)
    else:
        with pipeline:
            run_processing_loop(pipeline, rgb_queue, depth_queue, confidence_queue, args)


if __name__ == "__main__":
    main()