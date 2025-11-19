# Transfer Learning-Based Beamforming for MIMO Antenna Arrays Using Pre-Trained Neural Network Models
*A MATLAB–based deep learning framework for intelligent transmit beamforming.*

---

## 📘 Project Summary

This repository presents a complete end‑to‑end pipeline for **adaptive MIMO beamforming** using **transfer learning** with the pre‑trained **ResNet‑50** model.  
The system converts either **beam pattern images** or **user‑provided complex channel matrices** into a suitable input representation for a convolutional neural network, enabling accurate prediction of **complex antenna weights** for transmit beamforming.

Developed by:  
**Ayush Wattamwar**  
ECE, VIT Vellore

---

## 📄 Abstract

This paper presents a novel approach to adaptive antenna beamforming for MIMO wireless communication systems by leveraging transfer learning with pre-trained deep convolutional neural networks, specifically ResNet-50. Traditional beamforming methods are highly dependent on lengthy training procedures or extensive datasets, which limits adaptability in dynamic environments. Our method encodes channel state information and array parameters as 2D image-like matrices suitable for CNN input, enabling the repurposing of vision-trained neural networks for radio frequency beamforming tasks. Using MATLAB’s Deep Learning and Phased Array System Toolboxes, the proposed system fine-tunes ResNet-50 to predict optimal beamforming weights from minimal data, achieving rapid and robust beam pattern synthesis. Numerical simulations demonstrate superior performance in beamforming gain, signal-to-interference-plus-noise ratio (SINR), and computational efficiency compared to classical and baseline machine learning techniques. This work highlights the potential of cross-domain transfer learning for efficient, real-time beamforming in upcoming 5G/6G systems, aligning with industry demands for scalable, intelligent wireless infrastructure.

---

## 📑 Table of Contents
1. [Repository Structure](#repository-structure)  
2. [Methodology](#methodology)  
3. [How to Run](#how-to-run)  
4. [Technologies Used](#technologies-used)  
5. [Limitations & Future Work](#limitations--future-work)  
6. [Report](#report)

---

## 📂 Repository Structure

```
resnet-beamforming
│
├── training/
│   ├── step1.m
│   ├── step2.m
│   ├── step3.m
│
├── evaluation/
│   ├── step4.m
│   ├── step5.m
│
├── user_input_demo/
│   ├── user_input.m
│
├── dataset/
│   ├── BeamPatterns/
│   ├── beam_targets.mat
│
├── results/
│   ├── dataset_samples/
│   ├── training_plots/
│   ├── evaluation_plots/
│   ├── user_comparison_plots/
│
└── README.md
```

---

## 🧠 Methodology

### **1️⃣ Dataset Generation**
Beam patterns are synthetically generated using an 8‑element ULA.  
<img width="224" height="224" alt="pattern_0018" src="https://github.com/user-attachments/assets/c86738a4-27d8-4b30-8795-8eb05b2a00e0" />
 
`![Dataset Sample](results/dataset_samples/placeholder.png)`

---

### **2️⃣ Model Adaptation (ResNet‑50 for Regression)**  
The ImageNet ResNet‑50 is modified by:  
- Removing the classification head  
- Adding a fully connected regression layer (`2 × nTx` outputs)  
- Freezing early layers and fine‑tuning deeper layers  
<img width="2332" height="1248" alt="image" src="https://github.com/user-attachments/assets/821cea2f-89db-4200-a94a-6ea97c281801" />
`![ResNet Architecture](results/training_plots/model_graph.png)`

---

### **3️⃣ Training**
- Adam optimizer  
- Minor augmentation  
- 80/20 train‑validation split  

<img width="1720" height="880" alt="step3_output" src="https://github.com/user-attachments/assets/6f05a0be-fe1b-4e31-b0cd-60cd61ce9f80" />
`![Training Progress](results/training_plots/training_loss.png)`
<img width="1042" height="660" alt="step3_epochs_table" src="https://github.com/user-attachments/assets/4d59cc85-bbce-4b63-a022-34423ac1643f" />

---

### **4️⃣ Evaluation**
- RMSE & MAE of predicted weights  
- Beam pattern reconstruction  
- Comparison with MVDR, LMS, RLS  

<img width="742" height="387" alt="mvdr_optimal_parameters" src="https://github.com/user-attachments/assets/c733d061-0645-4891-8873-8ea716c293e0" />
<img width="726" height="367" alt="lms_optimal_parameters" src="https://github.com/user-attachments/assets/d297bf7a-d3fe-4f7c-91ea-9562c5147b5a" />
<img width="732" height="257" alt="rls_optimal_parameters" src="https://github.com/user-attachments/assets/5ca1a86b-bda0-4cf7-b5c5-892f193c3e2d" />
`![Evaluation](results/evaluation_plots/placeholder.png)`

---

### **5️⃣ User Input Demo**
Users can enter a **complex channel matrix H**, which is converted into a grayscale input to the neural network.  
The output weights are compared with classical algorithms.

<img width="719" height="172" alt="user_input_data" src="https://github.com/user-attachments/assets/86e55585-5139-461b-911b-69e9f6cf8ca5" />
<img width="224" height="224" alt="user_input_resnet_ready_gray" src="https://github.com/user-attachments/assets/4bfc1a5e-12d7-4308-9d24-f6ebbe6ba80a" />

`![User Demo](results/user_comparison_plots/placeholder.png)`

---

## 🛠️ Technologies Used
- **MATLAB**
- **Phased Array System Toolbox**
- **Signal Processing Toolbox**
- **Deep Learning Toolbox**

---

## 🔮 Limitations & Future Work

This work demonstrates the effective use of a fixed transmit antenna array size (**nTx = 8**) to achieve a strong balance between beamforming resolution and model complexity, enabling successful training and accurate prediction of beamforming weights using deep learning. Choosing 8 elements provided sufficiently narrow beam patterns and rich feature representations while keeping the model size and dataset size manageable.

However, this introduces a limitation:  
➡️ the current model and dataset are tailored **specifically** for 8‑element arrays.  
The architecture cannot directly support different array sizes without retraining or modifying the output structure.

Future extensions may include:  
- Generalized models supporting **variable antenna sizes**  
- Multi‑resolution or scalable architectures  
- Dynamic output layers adapting to different ULA configurations  

This design choice establishes a strong foundation for intelligent beamforming research while revealing promising pathways for future improvements.

---

## 🚀 How to Run

### **1. Generate dataset**
```matlab
step1.m
```

### **2. Modify ResNet‑50**
```matlab
step2.m
```

### **3. Train**
```matlab
step3.m
```

### **4. Evaluate**
```matlab
step4.m
```

### **5. Deployment**
```matlab
step5.m
```

### **6. User Input Beamforming**
```matlab
user_input_.m
```



## © Author

**Ayush Wattamwar**  
ECE, VIT Vellore

