# HerbAgent — AI-Powered Framework for Traditional Chinese Medicine Network Pharmacology  

*“HerbAgent is an AI multi-agent research framework designed to automate and enhance workflows in Traditional Chinese Medicine (TCM) network pharmacology.”*  

HerbAgent aims to assist researchers in TCM by integrating **Large Language Models (LLMs)** and **Multi-Agent Systems** to automate repetitive data processing, hypothesis generation, and network analysis — allowing scientists to focus more on discovery rather than manual tasks.


 📺 [Watch the HerbAgent Demo on YouTube](https://www.youtube.com/watch?v=M2J1N40AdJkx)
---

## Table of Contents

1. Project Background & Objectives  
2. Core Concepts & Architecture  
3. Main Functional Modules  
4. Technical Implementation Details  
5. Installation & Usage Guide  
6. Case Study / Example Use Cases  
7. Limitations, Challenges & Future Plans  
8. Contribution Guide & Community Support  
9. License & Acknowledgments  

---

## 1. Project Background & Objectives

### Background  
Network pharmacology has become a key method for modernizing and mechanistically studying Traditional Chinese Medicine (TCM).  
However, the research process — from herb-component-target-disease data collection, relation building, to network analysis (e.g., PPI, Random Walk with Restart) — is often **tedious and time-consuming**, requiring extensive manual effort.  

While there are databases and partial tools available, **no end-to-end, AI-driven system** currently exists to help generate mechanistic hypotheses or integrate knowledge from multiple biomedical sources automatically.

### Objectives  

HerbAgent is built with the following goals:  

- **Automation**: Automate the full research workflow — from inputting herbs/formulas to target prediction, network construction, and report generation.  
- **Intelligent Assistance**: Utilize LLM-powered agents to assist hypothesis generation, reasoning, and literature extraction.  
- **Modularity & Extensibility**: Each module (target prediction, network analysis, visualization, etc.) is modular and easy to replace.  
- **Explainability & Traceability**: Provide interpretable intermediate results and reasoning chains.  
- **Generalizability**: Although focused on TCM, the architecture can be extended to Western medicine or systems biology applications.  

---

## 2. Core Concepts & Architecture

<img width="2228" height="435" alt="image" src="https://github.com/user-attachments/assets/e634e27f-1fe4-4b9f-833e-595ebec76ac6" />

### Multi-Agent Architecture  

HerbAgent employs a **multi-agent system**, where each agent is responsible for a specific subtask in the research workflow:  

- **Herb–Target Agent**: Predicts potential compounds and molecular targets for given herbs or formulas.  
- **Syndrome–Disease Agent**: Maps TCM syndromes to modern biomedical diseases.  
- **Network Analysis Agent**: Builds the compound-target-disease network and performs network-level analysis (e.g., PPI, RWR, topology).  
- **Hypothesis / Report Agent**: Synthesizes all results into mechanistic hypotheses and generates scientific-style reports.  

This modular division ensures clarity, maintainability, and extensibility.

<img width="2186" height="1213" alt="image" src="https://github.com/user-attachments/assets/9dc1e9fb-8fc9-4b31-8e1a-70e7ee5998a3" />

### Data Flow & Control Flow  

1. **Input Stage** – User inputs herbs/formulas, syndromes, or diseases.  
2. **Agent Collaboration** – Each agent works sequentially or concurrently, exchanging intermediate outputs.  
3. **Network & Analysis Stage** – Constructs and analyzes networks using graph algorithms and biological data.  
4. **Report Generation** – A Report Agent compiles results into readable summaries, explanations, and visualizations.  
5. **Human-in-the-loop** – Users can refine or query agents interactively, forming a feedback loop.

<img width="1426" height="1189" alt="image" src="https://github.com/user-attachments/assets/3f6fa150-799d-4324-b25a-f3620bca9bc2" />


### Extensibility  

Each module follows a plug-and-play design, allowing researchers to replace or extend them easily.  
Adapters and hooks can be used to integrate new models, APIs, or visualization components without breaking the existing pipeline.

---

## 3. Main Functional Modules

| Module / Agent | Function | Input | Output | Key Techniques |
|-----------------|-----------|--------|---------|----------------|
| **Herb–Target Agent** | Predicts chemical compounds and potential targets for given herbs/formulas | Herb names, databases, or literature | Herb → Compound → Target mapping | Integrates TCMSP, PubChem, PharmMapper, or uses LLM-based extraction |
| **Syndrome–Disease Agent** | Maps TCM syndromes to modern diseases | Syndrome & disease names, corpus | Syndrome ↔ Disease associations | Text mining + entity linking with LLM inference |
| **Network Analysis Agent** | Constructs and analyzes the compound–target–disease network | Nodes (compounds, targets, diseases), edges | Graph, key nodes, ranked pathways | Graph theory via NetworkX / igraph, performs PPI, RWR, and topology analysis |
| **Hypothesis / Report Agent** | Generates hypothesis and report from all intermediate results | Outputs from other agents | Structured report + visualizations | LLM summarization + templated report generation |
| **Frontend / Interface Module** | Interactive visualization and user control | User input | Visualization & feedback UI | Streamlit / Gradio / Flask / Jupyter integration |

Supporting modules include:

- `target_downloader.py`: Target data fetching & caching  
- `symMap.py`: Syndrome–disease mapping utilities  
- `model_12_final.py`: Core orchestrator of the agent workflow  
- `web_interface.py`: Web API interface  
- `test_target_downloader.py`: Unit tests  
- `templates/`, `data/`, `downloads/`, `output/`: Storage and reporting directories  

---

## 4. Technical Implementation Details

### Compound & Target Prediction  

- Integrates public APIs and databases (TCMSP, UniProt, PubChem, DrugBank).  
- LLM-based text extraction for target mentions in literature.  
- Confidence scoring and filtering for prediction validation.  

### Network Construction & Analysis  

- Multi-type graph: nodes represent compounds, targets, diseases, syndromes.  
- PPI data integration (STRING, BioGRID).  
- **RWR (Random Walk with Restart)** algorithm for network diffusion.  
- Topological metrics: degree, betweenness, closeness, clustering coefficient.  
- Pathway extraction (shortest-path, k-step paths, flow-based).  

### Report & Visualization  

- Exports network to `.graphml`, `.gexf`, or JSON for Cytoscape / D3.js.  
- Visualization via Plotly, D3.js, or Cytoscape.js.  
- LLM-generated text explanations for hypothesis and mechanism interpretation.  
- Structured reports with figures and summaries in Markdown / HTML / PDF.  

### Workflow Orchestration  

- Central controller coordinates all agents sequentially or asynchronously.  
- Logs every process step for reproducibility and debugging (`herb_agent.log`).  
- Exception handling and checkpointing for robustness.  

### Performance & Scalability  

- Caching intermediate computations (targets, networks).  
- Parallel computation for large graphs.  
- API call optimization for cost and latency management.  

---

## 5. Installation & Usage Guide

### Environment Setup  

```bash
git clone https://github.com/zkManuel0123/HerbAgent.git
cd HerbAgent
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Configuration  

- Configure API keys (OpenAI, database APIs, etc.) in environment variables or a `config.yaml`.  
- Adjust parameters (e.g., restart probability, thresholds, paths) in the config file.  

### Command-Line Usage  

```bash
python model_12_final.py --herb "Astragalus membranaceus" --disease "Diabetes" --output_dir output/
```

### Python API  

```python
from model_12_final import HerbAgentMain

agent = HerbAgentMain(config_path="config.yaml")
result = agent.run(herb="Astragalus membranaceus", disease="Diabetes")
print(result.report_path)
```

### Web Interface  

```bash
python web_interface.py --host 0.0.0.0 --port 8080
```

Visit: [http://localhost:8080](http://localhost:8080)

### Outputs  

- **Report**: Hypothesis, explanations, and visualizations (`output/report_*.html` or `.pdf`)  
- **Network Files**: `.gexf`, `.graphml`, `.json`  
- **Intermediate Data**: compound-target tables, PPI subgraphs  
- **Logs**: `herb_agent.log`  

---

## 6. Case Study / Example

### Example: *Astragalus membranaceus* and Diabetes  

1. Input herb “Astragalus membranaceus” and disease “Diabetes”.  
2. **Herb–Target Agent** retrieves compounds (e.g., flavonoids, saponins) and predicts potential protein targets.  
3. Build a **compound–target–disease** network.  
4. Apply **Random Walk with Restart (RWR)** to identify diabetes-related targets.  
5. Extract potential mechanistic pathways connecting compounds and disease nodes.  
6. **Report Agent** generates hypothesis such as:  
   *“Astragalus may exert antidiabetic effects by modulating PI3K/Akt signaling pathways through compound X interacting with protein Y.”*  
7. Final output includes a full HTML/PDF report with visualized networks and ranked pathways.

---

## 7. Limitations, Challenges & Future Plans

### Current Limitations  

- **Data dependency**: Accuracy relies heavily on public databases.  
- **Prediction noise**: Some targets may be false positives, requiring manual curation.  
- **Scalability**: Large networks may cause high computational costs.  
- **Interpretability**: LLM-generated hypotheses may not always be scientifically rigorous.  
- **Limited feedback loop**: Lack of user–agent learning or adaptive refinement.  
- **Visualization**: Static reports may not provide deep interactivity.  

### Future Directions  

- Integrate omics and clinical datasets to increase biological relevance.  
- Introduce reinforcement learning for adaptive hypothesis refinement.  
- Support distributed computation for large-scale networks.  
- Add interactive, dynamic web visualization (zoom, highlight, filtering).  
- Extend to Western drugs, microbiome, or host–pathogen interactions.  
- Provide plugin APIs for external developers to add new agents.  
- Develop evaluation and benchmarking modules using known drug–target–disease triplets.  

---

## 8. Contribution Guide & Community

### How to Contribute  

1. Fork the repository  
2. Create your branch: `git checkout -b feature/your-feature`  
3. Implement, test, and document your feature  
4. Submit a Pull Request with clear descriptions  

### Reporting Issues  

Please use GitHub **Issues** to report bugs or suggest features.  
Include logs, inputs, expected vs. actual outputs whenever possible.

### Code Style  

- Follow **PEP8** conventions and use **Black / Flake8** for formatting  
- Include docstrings and inline comments  
- Add unit tests for critical modules (`pytest` recommended)  

### Community  

- Discussions, Q&A, and collaboration are welcome!  
- You may open a **Discussion** tab or a community chat group for support.  

---

## 9. License & Acknowledgments

### License  

This project is released under the **MIT License**.  
See the [LICENSE](./LICENSE) file for full details.

### Acknowledgments  

We would like to thank:  

- **LLM providers** such as OpenAI and Anthropic  
- **Databases**: TCMSP, PubChem, UniProt, STRING, DrugBank  
- **Libraries**: NetworkX, igraph, Plotly, Cytoscape  
- **Contributors & Testers** who supported and improved the project

📚 Citation

If you use HerbAgent in your research or project, please cite it as follows:
```
@misc{HerbAgent2025,
  author       = {Zhao, K.},
  title        = {HerbAgent: An AI-Powered Framework for Traditional Chinese Medicine Network Pharmacology},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/zkManuel0123/HerbAgent}},
}
```

---

## Summary

HerbAgent provides an **AI-powered, modular, and interpretable platform** for TCM network pharmacology research.  
It bridges **traditional herbal knowledge** with **modern computational systems pharmacology**, helping researchers discover potential mechanisms more efficiently.  

You can start right away with:

```bash
python model_12_final.py --herb "Astragalus membranaceus" --disease "Diabetes"
```

and receive a fully automated hypothesis report within minutes.  

---

> 🌿 *HerbAgent — Bringing AI-driven intelligence to Traditional Chinese Medicine research.*
