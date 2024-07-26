# EXplainable_AI
Explainable Artificial Intelligence (XAI)

Three versions are under development:

- FAR-Explainer: Feature Attribution using Ranking (Ft_Att_Rank.ipynb). In this version, we use a concept similar to Breiman's Feature Importance and SHAP, verifying the sensitivity of a trained model by "omitting" features. Then we build a transition matrix by omitting one feature (i) and two features (ij), and through this matrix, we apply Markovian modeling to get the stationary distribution that bears the feature importances. However, stationary distribution only holds positive values. Then, we must determine the direction of individual importances (positive and negative) relative to each data instance. (DISCONTINUED)

- SVD-Explainer: Jacobian-Singular Value Decomposition Feature Attribution (Taylor_Explainer.ipynb). In this version, we determine the Jacobian matrix (linear transformation) of M (trained m-class classification model) using the Finite Difference Method for systems of nonlinear equations [M(x + h) ~ M(x) + JM(x).h], x ∈ X is an instance in R^n and M: X -> Ω ⊂ R^m. Then, we decompose the Jacobian using SVD method returning [U dot S dot VT]. The vectors v_i give rise to an orthonormal basis for a j-dimensional subspace in R^n, j = min{m, n}. Denoting the coordinates of each basis vector v_i as (v_1i, ..., v_ni), the value v_ji can be interpreted as the importance of attribute j for v_i. Then we can measure the degree of importance of each attribute in the mapping by properly weighting their contribution according to phi_i = sum_{k=1 to j}[(sk/s1)*(vk.T dot x)*(vik)]. (WIP)

- T-Explainer: Taylor expansion-based technique (Taylor_Explainer.ipynb). Let X be a multidimensional dataset where each x = (x1, ..., xn) ∈ X is a data instance in Rn, and f be a binary machine learning model trained on X, that is, f(x) ∈ [0, 1]. The model f can be seen as a real-valued function f: X → Ω ⊂ R where Ω = [0, 1]. As a real function, f can be linearly approximated through first-order Taylor's expansion f(x + h) ≈ f(x) + ∇f(x) · h where h is a displacement vector corresponding to small neighborhood perturbation of x and ∇f(x), is the gradient (linear transformation) of f in x. The gradient of f in x corresponds, in fact, to the Jacobian matrix when f is a real-valued function. The dot product between the gradient and the displacement vector h is a linear map from Rn to R, being the best linear approximation of f near x. By definition, the T-Explainer is an additive feature attribution model  meaning that the importance value attributed to each feature can reconstruct the model prediction by summating these importance values. In T-Explainer, the feature attribution φi has a simple and intuitive geometric interpretation corresponding to the  ∇f(x) projection on the i-th feature axis.


We also included benchmarking tools in our framework (XAI_stability_metrics.ipynb and Taylor_Explainer.ipynb). Synthetic and real datasets, and five quantitative metrics:

- Relative Input/Output Stability (RIS and ROS): these metrics are used to evaluate explanation stability as to changes (local perturbations) in input data and output prediction probabilities, respectively (XAI_stability_faithfulness_metrics
.ipynb).
  
- Run Explanation Stability (RES): this metric assesses the consistency of several explanations for the same instance under the same settings, with higher values indicating lower stability rates (XAI_stability_faithfulness_metrics
.ipynb).
  
- Local Accuracy Preservation (LAP): this metric assesses the rates an additive feature importance explainer preserves its primary property of local accuracy, i.e., the model prediction should be reconstructed by the summation of the importance values. The ratio indicates the explanations' share, which preserves local accuracy (XAI_stability_faithfulness_metrics
.ipynb).

- Prediction Gap on Important Features (PGI): this metric assesses explanations' faithfulness by examining the impact of keeping important features and perturbing (deleting) non-important features on the model's predictions (XAI_stability_faithfulness_metrics
.ipynb).


# Cite Us

If you find our work useful, please cite our papers:

```
@article{ortigossa2024texplainer,
    title={{T-Explainer}: A Model-Agnostic Explainability Framework Based on Gradients}, 
    author={Ortigossa, Evandro S and Dias, F{\'a}bio F and Barr, Brian and Silva, Claudio T and Nonato, Luis Gustavo},
    journal= {Preprint arXiv:2404.16495},
    year={2024}
}

@article{ortigossa2024explainable,
    author= {Ortigossa, Evandro S and Gon{\c{c}}alves, Thales and Nonato, Luis Gustavo},
    journal= {IEEE Access}, 
    title= {{EXplainable} Artificial Intelligence ({XAI})--{From} Theory to Methods and Applications}, 
    year= {2024},
    volume= {12},
    pages= {80799-80846}
}
```
