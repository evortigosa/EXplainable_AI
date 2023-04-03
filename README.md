# EXplainable_AI
Explainable Artificial Intelligence (XAI)

Two versions under development:

- FAR Explainer: Feature Attribution using Ranking (Ft_Att_Rank.ipynb). In this version, we use a concept similar to Breiman's Feature Importance and SHAP, verifying the sensitivity of a trained model by "omitting" features. Then we build a transition matrix by omitting one feature (i) and two features (ij), and through this matrix, we apply Markovian modeling to get the stationary distribution that bears the feature importances. However, stationary distribution only holds positive values. Then we must determine the direction of individual importances (positive and negative) relative to each data instance.

- J-SVD Explainer: Jacobian-Singular Value Decomposition Feature Attribution (SVD_Explainer.ipynb). In this version, we determine the Jacobian matrix (linear transformation) of M (trained m-class classification model) using the Finite Difference Method for systems of nonlinear equations [M(x + h) ~ M(x) + JM(x).h]. Then we decompose the Jacobian using SVD method returning [U dot S dot VT]. The vectors v_i give rise to an orthonormal basis for a j-dimensional subspace in R^n, j = min{m, n}. Denoting the coordinates of each basis vector v_i as (v_1i, ..., v_ni ), the value v_ji can be interpreted as the importance of attribute j for v_i. Then we can measure the degree of importance of each attribute in the mapping by properly weighting their contribution according to phi_i = sum_{k=1 to j}[(sk/s1)*(vk.T dot x)*(vik)].
