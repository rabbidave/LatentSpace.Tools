# [Latent Space Tools](https://youtu.be/t9cYCfqrPJw)
Observability & Security for LLMs x Latent Space Apps

The following Software Suite, Visual Artifacts, and any/all concepts herein (hereafter Latent Space Tools) are made available under the Apache 2 license.

[Latent Space Tools](https://www.latentspace.tools) help conceptualize, visualize, and subsequently operationalize the necessary architecture and software components for secure LLM Deployment & Monitoring.

## Architectureal Overview:

<img src="https://github.com/rabbidave/Latent-Space-Tools/blob/main/a16zSummary.png" alt="Overview" title="Overview" width="70%">

[Click here for Detailed Annotations](https://github.com/rabbidave/Latent-Space-Tools/blob/main/a16zDetailAnnotated.pdf)

## Key Components: 

### Input Pre-Processing

[1) Prompt Injection Detection & Mitigation](https://github.com/rabbidave/Denzel-Crocker-Hunting-For-Fairly-Odd-Prompts)

[2) Service Denial & Performance Monitoring](https://github.com/rabbidave/StoopKid-Event-Driven-Input-Monitoring-for-Language-Models)

### Data Enrichment, Monitoring & Clustering
[3) Topic/Sentiment Modeling x Vector Comparisions & Cluster Defitntion](https://github.com/rabbidave/Jimmy-Neutron-and-Serverless-Stepwise-Latent-Space-Monitoring)

### Output Post-Processing
[4) Attack Mitigation, Appending (Un)Certainty & Response Non-Conformity](https://github.com/rabbidave/Squidward-Tentacles-and-Spying-on-Outputs-via-Conformal-Prediction)

### Output Forecasting
[5) Heatmaps x Dimensionality Drift via Conformal Prediction Intervals](https://github.com/rabbidave/Eliza-Thornberry-and-the-conformal-prediction-of-LLM-Behavior)



## Core Concepts:
### N-Dimensional Drift:
Given a [latent space](https://en.wikipedia.org/wiki/Latent_space) generally represents a reduced dimensionality from the feature space, we expect the 'aggregate' dimensions to be noisier than their components.

That said, the chosen dimensions should represent meaningful metrics worth monitoring; hence the value in conceptualizing, monitoring, and forecasting changes to those values

### Conformal Prediction

Latent Space Tools extensively leverage the concept of [conformal prediction](https://github.com/valeman/awesome-conformal-prediction); whereby previous outputs better predict future outputs than do Bayesian priors or assumptions

     