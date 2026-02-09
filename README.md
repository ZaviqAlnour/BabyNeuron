# Digital Baby: A First-Principles Cognitive Architecture

Digital Baby is an autopoietic learning entity designed from the ground up to mimic biological ontogeny. Unlike traditional AI models, it begins as a **tabula rasa** (blank slate) with no pre-trained weights, no static datasets, and no backpropagation. It learns exclusively through real-time sensory-motor interaction using bio-plausible mechanisms like Spiking Neural Networks (SNNs) and Predictive Coding.

## 👶 The Vision
The mission of this project is to engineer a system that generates its own intelligence through "lift"—continuous sensory-motor prediction error minimization. It is an architectural rebellion against the gravitational pull of disembodied, pre-trained intelligence.

## 🚀 Key Features
- **Tabula Rasa Genesis**: Zero pre-trained knowledge.
- **Active Sensorium**: Foveated event-based retina and cochlear filter bank.
- **Neuromorphic Core**: SNN with Spike-Timing-Dependent Plasticity (STDP).
- **Active Inference**: Cognition driven by the minimization of Expected Free Energy.
- **Homeostatic Regulation**: Simulates neuromodulators like Dopamine and Acetylcholine to regulate learning and focus.

## 🛠️ Tech Stack
- **Languages**: Python
- **Core Libraries**: `numpy`, `scipy`, `torch`, `snntorch`
- **Simulation**: Custom `WhiteRoomEnvironment`

## 🏃 Getting Started

### Prerequisites
Ensure you have Python installed. Install dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Simulation
To witness the "Digital Baby" in action as it begins to habitable its environment:
```bash
python scripts/run_simulation.py
```

### Running Tests
To verify the core and integration:
```bash
python -m unittest discover tests
```

## 📂 Project Structure
- `src/`: Core source code.
  - `sensory/`: Visual and audio encoding circuits.
  - `core/`: SNN, Predictive Coding, Memory, Homeostasis, and Action Selector.
  - `digital_baby.py`: The main agent class unifying all systems.
  - `environment.py`: Simulated physics and sensory environment.
- `scripts/`: Utility scripts for simulation and visualization.
- `tests/`: Unit and integration test suites.

---
*Crystallizing a new form of intelligence.*
