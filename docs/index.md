# Welcome to EdgeVolution Documentation

**Democratizing AI Optimization and Deployment on Microcontrollers**

---

## Overview

EdgeVolution is a comprehensive, open-source, end-to-end framework designed to simplify the complex process of optimizing and deploying artificial neural networks on resource-constrained edge devices, specifically microcontrollers (MCUs).

The rapid advancement of AI often outpaces its practical implementation on tiny, low-power devices due to significant technical hurdles. EdgeVolution bridges this gap by providing a unified pipeline that handles everything from dataset definition to multi-objective Neural Architecture Search (NAS) and direct deployment onto target hardware.

## The Challenge: AI on the Edge

Deploying effective AI on MCUs faces several critical constraints:

*   **Resource Limits:** MCUs have minimal memory (RAM/ROM) and computational power.
*   **Real-time Needs:** Many applications require low inference latency.
*   **Power Budgets:** Devices often need to run for extended periods on battery power.
*   **Deployment Complexity:** Adapting models for specific hardware and ensuring they actually *work* efficiently is often manual, error-prone, and requires deep expertise.
*   **Suboptimal Performance:** Using generic models or relying solely on predicted performance often leads to models that are inefficient or fail to meet hardware constraints.

## EdgeVolution's Key Contributions

EdgeVolution tackles these challenges through a unique, hardware-in-the-loop approach:

1.  **True Hardware-in-the-Loop (HWIL) Optimization:** Unlike many frameworks that rely on proxies or pre-computed tables, EdgeVolution *continuously deploys and measures* candidate neural network architectures directly on the target microcontroller **during** the optimization process. This provides accurate, real-world feedback on:
    *   **Inference Time (Latency)**
    *   **Energy Consumption**
    *   **Memory Usage (currently only ROM)**

2.  **Multi-Objective Neural Architecture Search (NAS):** Employs evolutionary algorithms (initially, a genetic algorithm) to automatically search for optimal neural network architectures, balancing:
    *   **Model Accuracy** (or other performance metrics)
    *   **Hardware Constraints** (Latency, Energy, Memory)

3.  **End-to-End, Automated Pipeline:** Streamlines the entire workflow:
    *   **Dataset Integration:** Flexible data loaders for various tasks (image classification, keyword spotting, sensor data analysis).
    *   **Integrated Preprocessing Search:** Can optimize preprocessing steps (like STFT parameters for audio) alongside the neural network architecture.
    *   **Automated Deployment:** Leverages the Zephyr RTOS and TensorFlow Lite Micro for seamless integration and deployment, generating functional binaries ready to run "out-of-the-box".

4.  **Flexibility and Adaptability:**
    *   **Data-Agnostic:** Easily adaptable to different datasets and classification tasks.
    *   **Hardware-Agnostic (within Zephyr):** Designed to support various MCUs compatible with the Zephyr RTOS.
    *   **Modular Design:** Allows for integration of different NAS strategies or hardware components.

5.  **Democratization and Reproducibility:** Aims to make advanced edge AI optimization accessible beyond specialized teams by providing an open-source, well-documented tool that improves performance and ensures reproducibility.

## Why Use EdgeVolution?

*   **Achieve True On-Device Performance:** Optimize models based on actual hardware measurements, not just estimations.
*   **Automate Complex Tasks:** Reduce the manual effort required for hardware-specific tuning and deployment.
*   **Navigate Trade-offs:** Automatically find architectures that optimally balance accuracy and resource constraints for your specific application and hardware.
*   **Ensure Deployability:** Implicitly filters out architectures that violate hardware limits (e.g., exceeding RAM).
*   **Improve Accessibility:** Lower the barrier to creating highly optimized edge AI applications.
*   **Open Source:** Free to use, modify, and contribute back to the community.

## Getting Started

Please use the [Getting Started](getting-started.md).

## Citation

If you use EdgeVolution in your research, please cite our paper:

tbd