# Knowledge and Data Jointly Driven Aeroengine Gas Path Performance Assessment Method

### Mechanistic Knowledge- and Data-Driven Decoupled Performance Assessment of Multiple Aeroengine Gas-Path Components

**English | [简体中文](README_CN.md)**

**Zhe Wang, Xuyun Fu, Rui Zhang, Zhengfeng Bai, Xiangzhao Xia, Wei Jiang**

*Chinese Journal of Aeronautics*, **37**(5), 533–557 (2024)

[Published Paper](https://doi.org/10.1016/j.cja.2023.09.011) | [DOI](https://doi.org/10.1016/j.cja.2023.09.011)

---

## 1. Overview

To address the challenges of **coupled performance degradation among multiple aeroengine gas-path components, complex mappings between monitored parameters and component health states, the dependence of conventional physics-based models on engine design parameters, and the limited physical interpretability of purely data-driven methods**, this study proposes a method for decoupled assessment of multiple gas-path components driven jointly by mechanistic knowledge and operational data.

First, aeroengine gas-path Fingerprint information is used to characterize the physical relationships between component performance degradation and monitored parameter responses, correct parameter deviations induced by non-component-degradation factors, and combine statistical screening with adaptive signal decomposition to suppress errors and extract degradation-related information from complex operational data.

On this basis, the performance degradation of individual gas-path components is introduced as **latent variables with explicit physical meaning**, forming an **asymmetric encoder–decoder framework incorporating mechanistic knowledge**. A data-driven nonlinear encoder learns the complex mapping between gas-path parameter deviations and multi-component degradation states. A physics-constrained decoder, constructed from the component degradation sensitivities represented by the Fingerprint, reconstructs gas-path parameter responses from the component degradation states. Reconstruction consistency and degradation regularization jointly guide latent-variable learning, enabling the joint identification and quantitative decoupling of performance degradation across multiple components.

Validation using real in-service aeroengine operational data and representative maintenance cases indicates that the method can reveal component performance evolution and reasonably characterize performance changes before and after maintenance. It provides technical support for integrating mechanistic knowledge and data in complex equipment modeling, interpretable health-state assessment, and refined operation and maintenance decision-making.

## 2. Core Idea

The central idea of this study is:

**Establish physical relationships through mechanistic knowledge, learn complex mappings through data-driven models, and guide the learning of physically meaningful latent variables through mechanistic reconstruction constraints to achieve decoupled multi-component performance degradation assessment.**

Unlike conventional physics-based analyses that depend entirely on high-fidelity gas-path thermodynamic models, or purely data-driven approaches without explicit physical constraints, this study treats **multi-component performance degradation as the key physical latent variables connecting mechanistic knowledge with monitoring data**. An asymmetric encoder–decoder structure establishes the relationship between gas-path parameter responses and component degradation states.

The framework integrates three mutually supporting mechanisms:

- **Data-driven Encoding:** The nonlinear representation capability of deep learning is used to extract degradation-related information from complex, coupled gas-path parameter deviation sequences and establish a mapping from monitored parameter responses to the performance degradation states of multiple components.
- **Physics-constrained Decoding:** A linear decoder grounded in the component degradation sensitivity relationships contained in the gas-path Fingerprint maps physically meaningful degradation latent variables back into the gas-path parameter space, enabling mechanistic reconstruction of monitored parameter responses.
- **Physics-guided Optimization:** Reconstruction consistency and component degradation constraints introduce physical relationships into model optimization, guiding the encoder to learn component degradation representations consistent with gas-path mechanistic knowledge.

Together, these mechanisms combine the nonlinear mapping capability of data-driven methods with the physical relationships provided by the Fingerprint. The latent variables thus represent component degradation states with explicit engineering meaning rather than merely abstract data features.

## 3. Methodology

The proposed knowledge- and data-driven gas-path performance assessment method comprises three main stages: **mechanistic knowledge-guided extraction of degradation information, nonlinear encoding with physically meaningful latent variables, and Fingerprint-constrained mechanistic decoding with joint optimization**.

<p align="center">
  <img src="figures/fig2_overall_framework.png" alt="Overall framework of the knowledge- and data-driven aeroengine gas-path performance assessment method" width="700" />
</p>
<p align="center"><em>Figure 2 in the published paper: Overall framework of the knowledge- and data-driven aeroengine gas-path performance assessment method.</em></p>

### 3.1 Mechanistic Knowledge-Guided Correction of Gas-Path Parameters and Extraction of Degradation Information

To address interference from non-component-degradation factors, outliers, and random noise in operational data, the method combines mechanistic knowledge with signal processing for data preprocessing.

The aeroengine gas-path Fingerprint characterizes the sensitivity relationships between component performance variations, non-degradation factors, and monitored gas-path parameter responses. As shown in Fig. 1, different component degradation and operating-related factors produce distinct response patterns in exhaust gas temperature (EGT), fuel flow (FF), and high-pressure rotor speed (N2). These physical relationships provide the mechanistic basis for gas-path parameter correction and Fingerprint-constrained reconstruction.

<p align="center">
  <img src="figures/fig1_fingerprint.png" alt="Aeroengine gas-path Fingerprint showing sensitivity responses in EGT, FF, and N2" width="850" />
</p>
<p align="center"><em>Figure 1 in the published paper: Fingerprint for a certain type of aeroengine (Wang et al., 2024).</em></p>

First, gas-path parameter shifts induced by non-component-degradation factors are corrected using the physical relationships between component degradation and monitored parameters represented by the aeroengine gas-path Fingerprint.

Next, the **Romanovsky statistical criterion** is used to identify and process gross errors in the monitored data. **Ensemble Empirical Mode Decomposition (EEMD) and Non-Local Means (NLM) filtering** are then combined to suppress noise in the complex operational data.

These procedures reduce the influence of non-degradation factors and measurement noise on performance assessment while extracting effective information related to gas-path component performance evolution.

### 3.2 Nonlinear Encoding and Physically Meaningful Latent Representation of Multi-Component Performance Degradation

To address the complex nonlinear coupling between gas-path parameter deviations and the performance degradation states of multiple components, a data-driven encoder is constructed using the **Dynamic Temporal Blended Network (DTBN) and a Multilayer Perceptron (MLP)**.

The DTBN integrates a Temporal Convolutional Network (TCN) and Long Short-Term Memory (LSTM) units to extract local temporal features and long-term dynamic dependencies, respectively. The MLP then learns the complex nonlinear relationships between monitored parameters and component performance degradation.

Unlike traditional autoencoders, in which the latent space primarily represents abstract data features, this study treats **the degradation of multiple gas-path components as latent variables with explicit physical meaning**, directly associating encoder outputs with the degradation states of individual components.

This design connects data-driven feature learning with engineering assessments of component health states, providing a basis for the joint identification of multi-component degradation.

### 3.3 Fingerprint-Based Mechanistic Decoding and Constrained Optimization

Using the component degradation sensitivity relationships in the gas-path Fingerprint, a **physics-constrained linear decoder** is constructed to reconstruct gas-path parameter responses from the component degradation latent variables produced by the encoder.

Specifically, the decoder uses the degradation states of individual components and their corresponding Fingerprint sensitivity weights to establish physical relationships between component degradation and monitored gas-path parameter deviations. The linear decoding structure described in the paper includes sensitivity weights and a bias term.

Two loss constraints are jointly introduced during model optimization:

- **Reconstruction Consistency:** Minimizes the discrepancy between gas-path parameter deviations reconstructed by the decoder and the actual input deviations, so that the encoder's learned latent variables are constrained by gas-path mechanistic relationships.
- **Degradation Regularization:** Uses the norms of Fingerprint sensitivity-weight vectors to constrain the degradation estimates of different components, mitigating the effects of differences in component sensitivities on model optimization.

The resulting **DTB-AutoEncoder** integrates mechanistic knowledge with data-driven learning for performance assessment.

Through nonlinear encoding, mechanistically constrained decoding, and joint optimization based on reconstruction consistency, coupled degradation information in gas-path measurements is mapped into physically meaningful multi-component degradation representations, enabling joint identification and quantitatively decoupled assessment of aeroengine gas-path component performance degradation.

## 4. Key Contributions

**(1) A mechanistic knowledge- and operational data-driven approach for multi-component gas-path performance assessment.**

To address the dependence of conventional physics-based models on engine design parameters and the limited interpretability of purely data-driven methods, the proposed approach combines gas-path Fingerprint knowledge with operational monitoring data. Parameter deviation correction, outlier suppression, and degradation information extraction provide a physically grounded data basis for assessing component performance degradation.

**(2) An asymmetric encoder–decoder model with component performance degradation as physically meaningful latent variables.**

Multi-component performance degradation is introduced as latent variables with explicit physical meaning. Data-driven nonlinear encoding learns the complex relationships between gas-path parameters and component degradation states, while a Fingerprint-based physics-constrained decoder reconstructs the monitored responses. Reconstruction consistency and degradation constraints jointly guide latent-variable learning to support the joint identification and quantitative decoupling of multi-component performance degradation.

**(3) Validation of engineering effectiveness using real in-service data and representative maintenance cases.**

The method is validated using real in-service aeroengine data and representative maintenance cases. Long-term component performance evolution and changes before and after maintenance are analyzed to evaluate the method's effectiveness in practical gas-path performance assessment, supporting interpretable health-state assessment and refined maintenance decisions.

## 5. Experimental Validation

The proposed knowledge- and data-driven gas-path performance assessment method is validated using real in-service aeroengine operational data and representative maintenance cases.

The experiments focus on the long-term evolution of multi-component performance degradation and changes in performance states before and after maintenance events, with the aim of examining the engineering consistency of the model's assessment results.

The following figures and tables are taken from the published journal article.

### 5.1 Engine `*991**`: Single-Overhaul Case

<p align="center">
  <img src="figures/fig18_engine_991_degradation.png" alt="Gas-path component performance degradation during the service life of engine 991" width="850" />
</p>
<p align="center"><em>Figure 18 in the published paper: Evolution of gas-path component performance degradation during the service life of engine <code>*991**</code>.</em></p>

<p align="center">
  <img src="figures/table12_engine_991_performance.png" alt="Multi-component performance assessment before and after overhaul for engine 991" width="850" />
</p>
<p align="center"><em>Table 12 in the published paper: Multi-component performance assessment results before and after overhaul for engine <code>*991**</code>.</em></p>

This case uses in-service degradation trajectories and the performance estimates before and after overhaul to analyze long-term component performance evolution and recovery following major repairs.

Before overhaul, the accumulated performance recession of the high-pressure turbine (HPT) relative to its factory condition was **0.6516%**. The estimated HPT performance indicator changed from **−0.1685%** before overhaul to **0.4272%** after major repairs, reflecting improved HPT performance following maintenance.

The value of 0.6516% denotes the accumulated recession relative to factory condition, not the amount of performance recovery achieved through overhaul.

### 5.2 Engine `*998**`: Two-Overhaul Case

<p align="center">
  <img src="figures/fig19_engine_998_degradation.png" alt="Gas-path component performance evolution across two overhaul cycles for engine 998" width="850" />
</p>
<p align="center"><em>Figure 19 in the published paper: Gas-path component performance evolution over two overhaul cycles for engine <code>*998**</code>.</em></p>

<p align="center">
  <img src="figures/table13_engine_998_performance.png" alt="Multi-component performance assessment over two overhaul cycles for engine 998" width="850" />
</p>
<p align="center"><em>Table 13 in the published paper: Multi-component performance assessment results over two overhaul cycles for engine <code>*998**</code>.</em></p>

This case covers two service and maintenance cycles, enabling analysis of the model's ability to characterize long-term performance degradation, differences among component states, and multiple maintenance events.

The paper reports an accumulated high-pressure turbine (HPT) performance recession of **0.6571%** before the first overhaul, consistent with oxidation and erosion of the HPT documented in the maintenance records.

During the second service interval, the accumulated performance recession of the low-pressure turbine (LPT) reached **0.6335%**, indicating relatively pronounced degradation.

These real in-service data and representative maintenance cases show that the proposed method can reasonably characterize component performance evolution and changes before and after maintenance, providing engineering support for multi-component degradation assessment.

## 6. Publication and Citation

**Title:**

Knowledge and data jointly driven aeroengine gas path performance assessment method

**Authors:**

Zhe Wang, Xuyun Fu, Rui Zhang, Zhengfeng Bai, Xiangzhao Xia, Wei Jiang

**Journal:** Chinese Journal of Aeronautics

**Year:** 2024

**Volume / Issue:** 37(5)

**Pages:** 533–557

**DOI:** [https://doi.org/10.1016/j.cja.2023.09.011](https://doi.org/10.1016/j.cja.2023.09.011)

**BibTeX:**

```bibtex
@article{Wang2024Knowledge,
  title   = {Knowledge and data jointly driven aeroengine gas path performance assessment method},
  author  = {Wang, Zhe and Fu, Xuyun and Zhang, Rui and Bai, Zhengfeng and Xia, Xiangzhao and Jiang, Wei},
  journal = {Chinese Journal of Aeronautics},
  volume  = {37},
  number  = {5},
  pages   = {533--557},
  year    = {2024},
  doi     = {10.1016/j.cja.2023.09.011}
}
```

## 7. Code and Data Availability

The real in-service aeroengine monitoring data and maintenance records used in the paper are not publicly distributed through this repository.

This repository contains an independent algorithmic implementation based on the research ideas presented in the paper. Its runnable demonstration uses synthetic data and does not constitute a reproduction of the paper's real in-service experimental results.

All published figures and tables retain their original source attribution and remain subject to the publisher's stated **CC BY-NC-ND 4.0** license and other applicable rights.

The article's open-access license does not automatically apply to the repository's source code or third-party data.
