# 🧠 Digital Baby: Arkitecture & Cognitive Engineering

## 🎯 Goal
The goal of the Digital Baby project is to create an **autopoietic learning system**—a cognitive architecture that "makes itself" through experience. We reject the "big data" paradigm and instead embrace **Digital Ontogeny**: the developmental history of an individual organism within its own lifetime.

## 🔭 Vision
We envision a future where intelligence is not a static blob of pre-computed weights, but a dynamic, pulsing process that is constantly predicting, acting, and learning from the physical (or simulated) world. Digital Baby is the first step toward **Antigravity Intelligence**: intelligence that generates its own lift from the sensorimotor stream, independent of historical archives.

---

## 🏗️ The Architecture

The system is organized into three major layers: the **Active Sensorium**, the **Neuromorphic Core**, and the **Active Control Loop**.

### 1. Active Sensorium
Instead of passive frames, we treat perception as an active process.
- **Foveated Retina**: Implements center-surround receptive fields that produce spikes only during change (ON/OFF). This mimics the sparse, event-based nature of biological vision.
- **Cochlear Filter Bank**: Decomposes sound waves into spectro-temporal patterns using Gammatone filters, mirroring the basilar membrane.

### 2. Neuromorphic Core
The "brain" of the Digital Baby.
- **SNN (Spiking Neural Network)**: The fundamental substrate. It uses **LIF (Leaky Integrate-and-Fire)** neurons.
- **STDP (Spike-Timing-Dependent Plasticity)**: The *only* learning rule. Weights change based on the temporal coincidence of pre- and post-synaptic spikes.
- **Predictive Coding Hierarchy**: A bio-plausible implementation of JEPA. Each layer predicts the activity of the layer below; only the **prediction error** is transmitted upward to update internal representations.

### 3. Active Control Loop
The "will" of the Digital Baby.
- **Active Inference**: The agent acts to minimize **Expected Free Energy (G)**. This balances exploitation (minimizing known errors) and exploration (seeking novelty to reduce future uncertainty).
- **Homeostasis**: A controller that simulates neuromodulators (Dopamine for surprise, Acetylcholine for precision/attention, Norepinephrine for arousal).

---

## 📂 File-by-File Breakdown

### `src/sensory/`
- **`spike_pattern.py`**: Defines the `SpikePattern` data structures. These are the "currency" of the system, carrying binary events and timestamps between modules.
- **`retina.py`**: The `FoveatedRetina` class. It converts RGB images into ON/OFF spikes using Difference of Gaussians (DoG) kernels and foveal magnification.
- **`cochlea.py`**: The `CochlearEncoder`. It uses a gammatone filter bank to turn audio into tonotopic spike patterns.

### `src/core/`
- **`snn.py`**: The `SpikingNeuralNetwork`. Implements the multi-layer spiking substrate and the STDP learning rule. It's where the raw sensory-motor associations are formed.
- **`predictive_coding.py`**: Hierarchical predictive coding engine. It performs inference to find the "hidden causes" of sensory inputs.
- **`homeostasis.py`**: The `HomeostaticController`. It tracks predictive accuracy and novelty to adjust global learning rates (via simulated Dopamine).
- **`memory.py`**: `AssociativeMemory` using Hopfield-like attractor dynamics. It binds simultaneous visual and audio patterns into stable concepts.
- **`action.py`**: The `ActionSelector`. Implements the Active Inference loop and learns a forward "mental model" of the world's transitions.

### `src/digital_baby.py`
The "Glue". It instantiates all the above modules and orchestrates the **life cycle**:
1. **Act**: Move gaze based on action selection.
2. **Sense**: Get spikes from retina/cochlea.
3. **Predict**: Compare spikes to top-down predictions.
4. **Learn**: Adjust weights via STDP and Hebbian rules.
5. **Regulate**: Update neuromodulators.

### `src/environment.py`
A minimal "White Room" environment. It provides a red cube for the agent to gaze at and simple audio cues to simulate name-object binding.

### `scripts/run_simulation.py`
A diagnostic script that runs the baby for 1,000 steps and tracks **Free Energy Habituation**. If the system is working, the Free Energy (surprise) should decrease over time as the baby "learns" the room.

### `tests/`
- **`test_core.py`**: Validates the SNN dynamics and Predictive Coding convergence.
- **`test_integration.py`**: Ensures the full `DigitalBaby` class can complete a sensorimotor cycle without crashing.

---
**Building intelligence that doesn't just process data, but inhabits it.**
