## 🧠 Tumor Image Segmentation using Region Splitting and Merging

This project implements **tumor image segmentation** using the **Region Splitting and Merging** technique, enhanced with a dynamic **Flask** web interface that allows user-controlled parameter tuning through sliders.

---

## 🚀 Features

- 📤 Upload medical images (e.g. MRI scans)
- 🎚️ Interactive sliders to control:
  - **Split Threshold**
  - **Merge Threshold**
  - **Minimum Region Size**
- ⚙️ Adjustable algorithm behavior through web UI
- 🧠 Suitable for tumor or anomaly segmentation tasks
- 🌐 Simple and clean Flask-based front-end

---

## 🖼️ Web Interface

The Flask app provides:
- A file upload form for image input
- Sliders to dynamically tune:
  - `Split Threshold`: Controls how sensitive the algorithm is to intensity variation during splitting.
  - `Merge Threshold`: Determines whether adjacent regions should be merged based on similarity.
  - `Minimum Region Size`: Avoids processing insignificant small regions.
- A “Submit” button that triggers segmentation and returns the processed image.
