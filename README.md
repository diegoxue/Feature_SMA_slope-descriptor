# slope-descriptor
Code for
Li, C., Liang, Q., Zhou, Y. & Xue, D. A knowledge‐based materials descriptor for compositional dependence of phase transformation in NiTi shape memory alloys. Materials Genome Engineering Advances (2024). https://doi.org:10.1002/mgea.72

1.0  slope-get, Extracting features from domain knowledge，Δτ;

1.5 data-get, Calculate the Δτ value for each component in the dataset;

2.0 multi Ap symbolic reg,  Implementing symbolic regression based on literature features and new feature Δτ about Ap data;

3.1.1 multi hp ridge reg polyK，3.1.2 multi hp ridge reg polyK without slope，3.2.1 multi enthalpy ridge reg polyK ，3.2.2 multi enthalpy ridge reg polyK without slope，3.3.1 multi hysteresis ridge reg polyK without slope , 3.3.2 multi hysteresis ridge reg polyK must slope, Implementing Kernel Ridge Regression Based on Literature Features and New Feature Δτ, including and excludingΔτ, about Ap、Enthalpy and Hysteresis data;

4 search, Search for high-performance alloys based on optimal models;

5 predict app, A simple prediction model for Ap, Enthalpy and Hysteresis of TiNi based shape memory alloys;

