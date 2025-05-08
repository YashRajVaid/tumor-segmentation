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
  
<img width="1318" alt="image" src="https://github.com/user-attachments/assets/7c9833c5-d833-4f01-aa97-b33aaa458062" />

<img width="1275" alt="image" src="https://github.com/user-attachments/assets/c8481cec-dc16-44f1-a59e-9b7dc4b1aade" />
