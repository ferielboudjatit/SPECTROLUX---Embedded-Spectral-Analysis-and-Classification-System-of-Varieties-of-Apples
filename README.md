# SPECTROLUX---Embedded-Spectral-Analysis-and-Classification-System-of-Varieties-of-Apples

## Overview

The **SPECTROLUX** project originates from an instrumentation initiative aimed at analyzing **absorption or reflection spectra** to extract information on the **chemical composition** and **physical properties** of studied materials.

An initial prototype was designed to observe spectral variations using a **Hamamatsu C12880MA mini-spectrometer** and an **STM32 microcontroller**.
These components offer compactness and reduced data size, enabling real-time processing and analysis of **288 spectral bands** ranging from **340 to 850 nm**, providing sufficient resolution for various analytical applications.

---

## Application — Fruit Variety Identification

The envisioned application was the **identification of apple varieties** 🍎 based on their **reflection spectra**.
Unlike traditional imaging, spectral analysis reveals **internal fruit characteristics invisible to the naked eye**.

To achieve this, a **1D Convolutional Neural Network (CNN)** model was trained and deployed on a **MAX78000 microcontroller**, responsible for executing the embedded AI model for apple variety classification.

### MAX78000 Main Features:

* Low-power processor optimized for **Machine Learning**
* **Hardware accelerator** for neural networks

Near-infrared (NIR) data is particularly useful in agriculture, as it provides information on **firmness**, **water content**, and **sugar levels (Brix)**, which are key indicators of fruit quality.
This spectral approach allows more accurate variety classification compared to simple image-based methods.

---

## Methodology 

### Instrumentation

* The **spectrometer** captures apple reflection spectra under illumination from a **halogen lamp**, chosen for its **continuous emission spectrum** extending from visible to near-infrared (unlike LEDs).
* A **STM32F303K8 microcontroller** was programmed to:

  1. Acquire reflection spectra from the Hamamatsu spectrometer
  2. Transmit data to a PC for visualization and storage
  3. Communicate with the **MAX78000**, which performs classification

---

### Database Construction

To train the classification models, we built a **spectral database** composed of **40 spectra per variety** (*CRIPPSRED, STORY, JULIET, and GALA*) measured from **4 apples per variety**.

Each apple was **cut in half**, and **5 spectra per half** were captured at different angles.
The distance between spectrometer and apple was kept constant to ensure consistent results.

Since calibration was done only once (on the first apple) and fruits have different volumes, **position markers** were drawn on the measurement base to maintain alignment during acquisition.

---

### Data Processing & Classification Models

After reviewing various preprocessing and classification techniques from literature (see Appendix 1), three models were tested for variety classification:

* **Support Vector Machine (SVM)**
* **Neural Network (NN)**
* **1D Convolutional Neural Network (CNN 1D)** : chosen to leverage MAX78000’s convolution accelerator and evaluate embedded performance

We also explored **Principal Component Analysis (PCA)** as a preprocessing step to reduce dimensionality while preserving essential information.
Models were compared **with and without PCA** to measure its impact on classification performance.


---

## Implementation on MAX78000

The implementation of the model offering the **highest accuracy** was studied in order to be **deployed on the MAX78000 board**, taking advantage of its **CNN accelerator**.

All the required steps, as well as the libraries to install, are detailed on the [AI8X Synthesis GitHub repository](https://github.com/analogdevicesinc/ai8x-synthesis/) (Analog Devices).
However, here is an overview of the essential elements to understand before starting this implementation.

With a dataset and model available, the **training**, **quantization**, and **evaluation** steps are performed iteratively until satisfactory results are achieved.

Once this stage is reached, the **quantized model**, optimized for the MAX78000, is used to **generate C code**.

This code allows the model’s **weights and architecture** to be loaded into the MAX78000 and to execute **inference directly on the device**.

To facilitate the training and synthesis of models compatible with the MAX78000, **dedicated environments** have been developed (**ai8x-training** and **ai8x-synthesis**) both based on the **ai8x library**, which itself relies on **PyTorch**.

This library provides a set of functions specifically designed to train models optimized for the MAX78000 hardware, directly integrating **quantization** and **evaluation** stages adapted to its CNN accelerator.
It is also used during the synthesis phase to **convert the trained model into a deployable version**, generating the necessary code for deployment and inference execution on the target board.

---

### Training

Training is performed within the **ai8x-training** Python environment.

From the `ai8x-training` directory, training is launched by executing the `train.sh` script.
This script calls the `train.py` file with a set of configuration parameters, including the model to train and the dataset to use:

```bash
./train.py --epochs 200 --deterministic --compress schedule.yaml \
--model ai85apple_discrimination --dataset AppleSpectra \
--confusion --param-hist --pr-curves --embedding \
--device MAX78000 "$@"
```

**--model**:
Defines the PyTorch model to be used.
The supported models are implemented in Python scripts located in the **models/** directory.
Each script exports a `models` variable that maps the value passed to `--model` to a specific PyTorch model implementation.

**--dataset**:
Specifies the dataset to use for training.
The data-loading scripts are located in the **datasets/** directory and export a `datasets` variable, mapping the provided value of `--dataset` to a dataset compatible with **torchvision**.
The dataset files are then downloaded into the **data/** subdirectory.

At the end of training, a **checkpoint file** named `best.pth.tar` is generated.
This file is stored within the **logs/** directory.

It is possible to use **Quantization Aware Training (QAT)**, which is **enabled by default**.
Alternatively, quantization can be skipped during training and performed later as **post-training quantization** by adding the parameter:

```bash
--qat-policy None
```

**Note:**
This development environment imposes a **limit of 1024 neurons** on the input layer of the linear (fully connected) layer.

---

### Quantization

This step is performed within the **ai8x-synthesis** Python environment.

In the case of **post-training quantization**, this process converts the floating-point weights extracted from the checkpoint file into **fixed-point weights** compatible with the MAX78000’s hardware CNN accelerator.

First, copy the file `best.pth.tar` from the directory `ai8x-training/logs/` to the directory `trained/`.
Then, execute the `quantize.sh` script, which calls `quantize.py` to perform quantization:

```bash
./quantize.py trained/best.pth.tar trained/best-q.pth.tar --device MAX78000 -v "$@"
```

This quantization step produces the file `trained/best-q.pth.tar`.
This file can then be used either for **evaluation** or to **generate C code** for execution on the MAX78000.

This step is also performed in the case of **QAT**, since the quantizer must always produce a checkpoint file that can be read by the **Network Loader**.

---

### Evaluation

This step is carried out within the **ai8x-training** Python environment.

Model evaluation occurs at two points in the workflow:

1. **After training**, and
2. **After quantization**.

It relies on **test data** to measure the model’s performance.
Evaluation after quantization allows the analysis of how weight quantization impacts the model’s accuracy.

Once the weights have been quantized, the results can be tested by using the `--evaluate` option of the `train.py` script.
The `train.py` script will then use the quantized weights generated by `quantize.py` to perform evaluation:

```bash
./train.py --model ai85apple_discrimination --dataset AppleSpectra \
--confusion --evaluate \
--exp-load-weights-from ../ai8x-synthesis/trained/best-q.pth.tar \
-8 --save-sample 4 --device MAX78000 "$@"
```

For the next step, it is necessary to add the option `--save-sample x`
(where `x` is an arbitrary index smaller than the batch size) to generate a file named `sample.npy`,
which saves an input sample from the training data.

---

### C Code Generation

This step is carried out in the **ai8x-synthesis** environment.

This operation is performed using the **Network Loader (`ai8xize.py`)**, the **checkpoint file**, and a **YAML description file** that defines the model within the context of the MAX78000’s CNN architecture.

The **YAML file** contains details of each model layer defined in the `forward()` function, such as operation type, kernel size, padding, activation function, etc.
It also includes MAX78000-specific parameters, such as the number of processors to activate for CNN operations.
For more details about processor configuration, see the documentation from [Analog Devices](https://www.analog.com/en/resources/media-center/videos/6313215449112.html) 

**Note:**
Some functions, such as `permute`, if used in the `forward()` function, are **not recognized** by the YAML file.
As an alternative, such operations can be handled in the **data loading script** instead of the model definition script.

The YAML file must be stored in the **networks/** subdirectory.

```bash
./ai8xize.py --verbose --log --test-dir demos/ \
--prefix ai8x-apple-discrimination \
--checkpoint-file trained/best-q.pth.tar \
--config-file networks/ai8x-apple_discrimination.yaml \
--softmax --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint
```

During this synthesis phase, several `.c` and `.h` files are generated, containing all the functions related to the CNN model, as well as a **Makefile**.

---

**Notes:**

* It is essential to verify that this **Makefile** correctly includes the paths to all directories containing the required libraries for compilation.
  To do this, the variables `VPATH` and `IPATH` should be used to indicate additional locations where the Makefile should look for **source files (.c)** and **header files (.h)** respectively.

For example, the following lines were added:

```bash
VPATH += /home/feriel/MaximSDK/Libraries/MiscDrivers/Display 
IPATH += $(CMSIS_ROOT)/Include  
```

* It is also important to ensure that the directories containing the required libraries have the **appropriate access permissions**.

---

## 📊 Results

### Database

| Figures                                                        |
| -------------------------------------------------------------- |
| ![Lamp Comparison](include/Exploration_lampes.png)             |
| ![Apple Varieties](include/Différentes variétés de pommes.png) |
| ![Story Apple](include/Pommes_Story.png)                       |
| ![Gala Apple](include/Pommes_Gala.png)                         |
| ![Juliet Apple](include/Pommes_Juliet.png)                     |
| ![Cripps Red Apple](include/Pommes_CrippsRed.png)              |

---

### CNN1D Model Architecture

| Layer         | Kernel  | Output Shape    |
| ------------- | ------- | --------------- |
| Input         | -       | None × 1 × 288  |
| Conv1+ReLU    | 5×1×32  | None × 32 × 288 |
| MaxPool1      | 2       | None × 32 × 144 |
| Conv2+ReLU    | 3×32×64 | None × 64 × 144 |
| MaxPool2      | 2       | None × 64 × 72  |
| Conv3+ReLU    | 3×64×14 | None × 14 × 72  |
| Flatten       | -       | None × 1008     |
| FC1+ReLU      | -       | None × 32       |
| Dropout (0.3) | -       | None × 32       |
| FC2           | -       | None × 4        |

---

### Neural Network (NN) Architecture

| Layer            | Units | Output Shape   |
| ---------------- | ----- | -------------- |
| Input            | -     | None × 1 × 288 |
| Dense1 (ReLU)    | 20    | None × 20      |
| Dense2 (ReLU)    | 30    | None × 30      |
| Dense3 (Softmax) | 4     | None × 4       |

---

### Classification Results

| PCA         | Set        | NN    | SVM | CNN1D  |
| ----------- | ---------- | ----- | --- | ------ |
| With PCA    | Train      | 100%  | -   | 96.55% |
|             | Validation | 75%   | -   | 67.86% |
|             | Test       | 75%   | 38% | 50%    |
| Without PCA | Train      | 100%  | -   | 85.34% |
|             | Validation | 62.5% | -   | 75%    |
|             | Test       | 62%   | 38% | 81.25% |

---

### Implementation on MAX78000

| Figures                                                    |
| ---------------------------------------------------------- |
| ![QAT Evaluation](include/Evaluation_result2.png)          |
| ![Code Generation](include/Generation_result.png)          |
| ![Board Implementation](include/Implementation_result.png) |


---

## Appendix 1 — Literature Review

| **Criterion**              | **Details**                                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Fruit storage**          | Stored at 4°C for 2–50 days, then kept at 20–25°C for 24h before measurements [Martinez 2010, Li 2018, Cen 2007, Cao 2010].                            |
| **Sample height**          | Optimal 100 mm with 20°–25° field of view [Shao 2008, Cen 2007].                                                                                       |
| **Light source**           | Halogen lamp, 14.5V–20V, 45° incidence angle [Shao 2008, Martinez 2010].                                                                               |
| **Averaging measurements** | Multiple points per fruit, 2–3 scans per point; mean used to reduce noise. Examples: 4 parts × 2 scans [Martinez 2010]; 150 scans per apple [Li 2018]. |
| **Sample size**            | 100 apples/variety [Li 2018]; 25 peaches/variety [Wu 2006].                                                                                            |

#### Preprocessing Methods

| **Method**                                  | **Description**                                                                          |
| ------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Spectral range limitation**               | Relevant range e.g., 400–1000 nm [Cen 2007, Cao 2010].                                   |
| **Sample organization**                     | Random smooth arrangement to balance fruit variability [Martinez 2010].                  |
| **Multiplicative Scatter Correction (MSC)** | Reduces additive/multiplicative effects [Li 2018, Wu 2006].                              |
| **Savitzky–Golay smoothing**                | Noise reduction preserving spectral features [Li 2018].                                  |
| **PCA + SPA**                               | PCA with Successive Projection Algorithm for wavelength selection [Cen 2007].            |
| **Genetic Algorithm (GA)**                  | Automatic wavelength selection, e.g., 4 wavelengths for grape discrimination [Cao 2010]. |

#### Classification Models

| **Model**                       | **Description**                                                                          |
| ------------------------------- | ---------------------------------------------------------------------------------------- |
| **Partial Least Squares (PLS)** | Robust predictive models for variety discrimination [Cen 2007].                          |
| **SVM**                         | Used for apple and grape variety discrimination; effective for small datasets [Li 2018]. |
| **Neural Networks (ELM, BPNN)** | PCA+ELM accuracy 96.67% [Li 2018]; good balance between complexity and performance.      |


---

## 🧩 Discussion

* The **1D CNN** outperforms other models in accuracy.
* **Dimensionality reduction** can benefit linear models like SVM but may cause significant information loss for CNNs.
* The **8-bit quantization** required for MAX78000 impacts model performance but greatly reduces energy and memory usage.
* Despite these limitations, the deployment validated the **complete embedded AI workflow**, from spectral acquisition to on-chip inference.

## 📚 Citation

If you use this work, please cite the following references and related studies:

```bibtex
@inproceedings{martinez2010non,
  title={Non-invasive estimation of firmness in apple fruit using VIS/NIR spectroscopy},
  author={Mart{\'\i}nez, M and Wulfsohn, D and Toldam-Andersen, T},
  booktitle={XXVIII International Horticultural Congress on Science and Horticulture for People (IHC2010): International Symposium on 934},
  pages={139--144},
  year={2010},
  doi = {10.17660/ActaHortic.2012.934.15}
}

@article{li2018apple,
  title={Apple variety identification using near-infrared spectroscopy},
  author={Li, Caihong and Li, Lingling and Wu, Yuan and Lu, Min and Yang, Yi and Li, Lian},
  journal={Journal of Spectroscopy},
  volume={2018},
  number={1},
  pages={6935197},
  year={2018},
  publisher={Wiley Online Library},
doi = {10.1155/2018/6935197}
}

@article{cen2007combination,
  title={Combination and comparison of multivariate analysis for the identification of orange varieties using visible and near infrared reflectance spectroscopy},
  author={Cen, Haiyan and He, Yong and Huang, Min},
  journal={European Food Research and Technology},
  volume={225},
  pages={699--705},
  year={2007},
  publisher={Springer},
doi = {10.1007/s00217-006-0470-2}
}

@article{cao2010soluble,
  title={Soluble solids content and pH prediction and varieties discrimination of grapes based on visible--near infrared spectroscopy},
  author={Cao, Fang and Wu, Di and He, Yong},
  journal={Computers and Electronics in Agriculture},
  volume={71},
  pages={S15--S18},
  year={2010},
  publisher={Elsevier},
  doi={10.1016/j.compag.2009.05.011}
}

@article{Shao11022008,
author = {Yongni Shao and Yong He},
title = {Nondestructive Measurement of Acidity of Strawberry Using Vis/NIR Spectroscopy},
journal = {International Journal of Food Properties},
volume = {11},
number = {1},
pages = {102--111},
year = {2008},
publisher = {Taylor \& Francis},
doi = {10.1080/10942910701257057}}

@inproceedings{wu2006fast,
  title={Fast discrimination of juicy peach varieties by Vis/NIR spectroscopy based on bayesian-sda and pca},
  author={Wu, Di and He, Yong and Bao, Yidan},
  booktitle={International Conference on Intelligent Computing},
  pages={931--936},
  year={2006},
  organization={Springer},
  doi ={10.1007/11816157_113}
}
