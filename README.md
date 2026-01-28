# YoableSeg

**YoableSeg** is an AI-powered image segmentation and annotation tool designed to make dataset labeling faster and more efficient. It implements **Grounding DINO** for text-prompted detection and **SAM3** for precise segmentation. YoableSeg provides an intuitive interface for managing images, running AI-assisted labeling, and exporting labels in a format compatible with machine learning models.


For non-WPF version you can build the legacy source or use v1.2.0 from releases - [Legacy branch](https://github.com/Babyhamsta/Yoable/tree/legacy).

<img width="1106" height="713" alt="image" src="https://github.com/user-attachments/assets/053aefe3-2188-4fb2-8050-f0bfcd61e2e7" />

## 🚀 Features

- **Text-Prompted Segmentation** - Uses **Grounding DINO** for natural language object detection ("car. tire. person.")
- **SAM3 Segmentation** - State-of-the-art instance segmentation with **Segment Anything Model 3**
- **Precise Polygon Masks** - Generates accurate segmentation masks, not just bounding boxes
- **Batch Processing** - Efficient batch inference for multiple objects in a single image
- **Smart Post-Processing** - Automatic NMS, overlap resolution, hole filling, and small region removal
- **Manual Labeling Tools** - Easily add, edit, and remove segmentation masks and bounding boxes
- **Bulk Image Import** - Load multiple images at once
- **YOLO Segment Format Support** - Import and export annotations in **YOLO segmentation format**
- **GPU Acceleration** - CUDA support for ultra-fast inference
- **Customizable UI** - Light/Dark theme and customizable label appearance
- **Crosshair Overlay** - Align annotations with precision
- **Adjustable AI Confidence** - Set detection confidence thresholds for better accuracy
- **Auto Updates** - Get the latest features and fixes with built-in update checks (Can be disabled via settings)
- **Project Support** - Create and save projects so you can pick back up where you left off

## 📥 Installation

1. Download the latest release from our [GitHub Releases](https://github.com/Babyhamsta/Yoable/releases)
2. Run YoableSeg (No install required!)
3. First-time setup will automatically install:
   - Python environment with PyTorch (CUDA 12.1)
   - **SAM3** (Segment Anything Model 3) from Meta
   - **Grounding DINO Base** for text-prompted detection
   - All required dependencies
4. Authenticate with HuggingFace (required for model downloads)

**Requirements:**
- Windows 10/11
- Python 3.10-3.13
- NVIDIA GPU with CUDA support (recommended for fast inference)
- ~5GB disk space for models

## 🛠️ How to Use

### Importing Images
- Click **"Import Image"** or **"Import Directory"** to load images.
- The images will appear in the **image list**.
- Use the scroll wheel to navigate through the imported images.
- Use Ctrl + Scroll Wheel to zoom in and out of an image.

### Applying Labels
- **Text-Prompted Auto-Labeling**: Enter natural language prompts (e.g., "car. person. dog.") and click **"Auto Label Images"**
  - Grounding DINO detects objects matching your text prompts
  - SAM3 generates precise segmentation masks automatically
- **Manual Labeling**: Use the drawing tools to create segmentation masks or bounding boxes
- **Box-to-Mask**: Draw a box and let SAM3 generate the precise mask
- **Point-to-Mask**: Click on an object to generate its mask

### Managing Labels
- Labels appear in the **label list**.
- Click on a label to edit it.
- Press **Delete** to remove selected labels.
- Use arrow keys for precise label movement.
- Use Ctrl + A to select all labels.
- Use Ctrl + Z to undo deletes, moves, and ect.
- Use Shift + Scroll Wheel to change classes on selected labels or while drawing a label.

### Importing & Exporting Labels
- **Import Labels**: Load existing YOLO segmentation format label files
- **Export Labels**: Save labeled data in YOLO segmentation format with polygon masks

### Updating YoableSeg
- YoableSeg automatically checks for updates
- If a new version is available, you'll be prompted to update

## 🔬 Technical Implementation

### Pipeline with SAM3

1. **Grounding DINO** (Text → Boxes)
   - Zero-shot object detection using natural language
   - Input: "car. tire. person."
   - Output: Bounding boxes with confidence scores

2. **Non-Maximum Suppression**
   - Removes duplicate/overlapping detections
   - Configurable IoU threshold

3. **SAM3 Batch Inference** (Boxes → Masks)
   - State-of-the-art segmentation from Meta
   - Ultra-fast batch processing
   - FP16 inference on CUDA GPUs

4. **Post-Processing Pipeline**
   - Small region removal
   - Morphological hole filling
   - Overlap resolution (priority by confidence)
   - Polygon simplification

5. **Vector Output**
   - Precise polygon contours
   - YOLO segmentation format
   - Efficient storage and editing

**Why SAM3 instead of SAM2?**
- SAM3 is Meta's latest segmentation model
- Improved accuracy and speed over SAM2
- Better handling of complex boundaries
- Native HuggingFace integration


## 🌍 Contributing
YoableSeg is **open-source**! Contribute by reporting issues, suggesting features, or improving the code.

## 📌 Support
For help and troubleshooting, visit our [GitHub Issues](https://github.com/Babyhamsta/Yoable/issues) or join our community.

---
⭐ **Star this repo** if you find it useful!
