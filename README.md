# parallel-cnn-training-cifar10


## 📌 Project Description

* 🔬 **Project Title:** A Comparative Study of Serial and Simulated Parallel CNN Training on CIFAR-10 Using TensorFlow
* 🧠 **Objective:** To evaluate the impact of parallelism on CNN training performance and automate the conversion of sequential training scripts using LLMs.

### ✅ Key Highlights

* 🖼️ **Dataset:** CIFAR-10 — 60,000 32x32 color images across 10 classes.
* 🧱 **Model:** Custom Convolutional Neural Network implemented in PyTorch.
* 🔁 **Serial Version:** Standard training script for CNN classification.
* 🚀 **Parallel Version:** Automatically generated using a custom LLM-powered tool that converts serial code into MPI-parallelized code using `mpi4py`.
* ⚙️ **Parallel Framework:** MPI simulation using `mpi4py`, allowing multi-process training across CPU cores.
* 📉 **Performance Metrics:**

  * Achieved **3.29× speedup** using 4 workers.
  * Maintained **82% parallel efficiency**.
  * Classification accuracy remained consistent across serial and parallel runs.
* 🛠️ **Automation Tool:** Developed a prototype LLM-based script that reads PyTorch training code and outputs MPI-compatible parallel code.
* 📊 **Evaluation:** Comparison based on training time, speedup, efficiency, and model performance across different worker counts (1, 2, 4).

### 🧪 Outcomes

* Demonstrated that simulated parallel training reduces CNN training time significantly with minimal code rewriting.
* Validated the feasibility of using large language models to automate parallelization tasks in deep learning workflows.

---

Let me know if you'd like to add a “How to Run” section or visual summary!
