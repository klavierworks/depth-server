# OAK-D Lite Depth Display (Python)

A simple Python script to display the depth feed from an OAK-D Lite camera.

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install depthai opencv-python numpy
```

### 2. Connect Your OAK-D Lite Camera

Connect your OAK-D Lite camera to your computer via USB-C.

## Usage

Simply run the script:

```bash
python oak_depth_display.py
```

Or make it executable and run directly:

```bash
chmod +x oak_depth_display.py
./oak_depth_display.py
```

## Controls

- **Press 'q'** to quit the application

## What You'll See

The script displays a color-mapped depth visualization where:
- **Red/Yellow/Warm colors** = Closer objects
- **Blue/Purple/Cool colors** = Farther objects
- **Black** = No depth data or out of range

## Features

- Real-time depth visualization at 400p resolution
- JET colormap for intuitive depth perception
- High-density stereo depth processing
- Left-right consistency check for better accuracy
- Depth aligned to RGB camera frame

## Troubleshooting

### Camera Not Detected
```bash
# List connected devices
python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

### Permission Issues (Linux)
If you get USB permission errors on Linux:
```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Installation Issues
If depthai installation fails, try:
```bash
pip install --upgrade pip
pip install depthai --upgrade
```

## Customization

You can modify the script to:
- Change resolution (THE_400_P, THE_480_P, THE_720_P, THE_800_P)
- Use different colormaps (COLORMAP_TURBO, COLORMAP_VIRIDIS, etc.)
- Add RGB camera feed alongside depth
- Save depth data to files
- Process depth data for specific applications

## System Requirements

- Python 3.7+
- USB 3.0 or USB-C port
- macOS, Linux, or Windows
- OAK-D Lite camera

## Notes

The OAK-D Lite uses:
- **CAM_B** - Left mono camera
- **CAM_C** - Right mono camera  
- **CAM_A** - RGB camera (used for alignment reference)

The stereo pair (CAM_B and CAM_C) generates the depth map.