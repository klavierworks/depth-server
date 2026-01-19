# Depth Camera filtered virtualcam

## Installation
Ensure OBS is installed. Open it once, start the virtual camera, close OBS. A virtual camera is now available for us to use, regardless of whether OBS is running.

To run depth filter:

* python3 -m venv depthenv
* source depthenv/bin/activate
* pip install -r requirements.txt
* python main.py

## Arguments
* --min_z = Minimum depth (Z) in millimeters (default: 500)
(default=500)
* --max_z = Maximum depth (Z) in millimeters (default: 5000)
(default=5000)
* --min_x = Minimum X coordinate as percentage 0-100, 0=left edge (default: 0)
(default=0.0)
* --max_x = Maximum X coordinate as percentage 0-100, 100=right edge (default: 100)
(default=100.0)
* --min_y = Minimum Y coordinate as percentage 0-100, 0=top edge (default: 0)
(default=0.0)
* --max_y = Maximum Y coordinate as percentage 0-100, 100=bottom edge (default: 100)
(default=100.0)
* --debug = Show debug info and boundary lines on screen
* --virtual-cam = Output to virtual webcam