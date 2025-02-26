##############################################################
#          XAI stability and faithfulness metrics 
# -- RIS, ROS, RES, PGI, LAP, and CAI (extensible versions)
# 
# For an overview of XAI metrics and evaluation, we refer to:
# http://dx.doi.org/10.1109/ACCESS.2024.3409843
# http://dx.doi.org/10.48550/arXiv.2404.16495
#
# TODO:
# - import T-Exp (making T-Exp as a Python module)
##############################################################

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import data_processing as dp
import xgboost as xgb
import warnings

import lime
import lime.lime_tabular
import shap

# from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from captum.attr import IntegratedGradients
from captum.attr import InputXGradient
from captum.attr import DeepLift
from captum.attr import LRP

from tqdm import tqdm
shap.initjs()


##############################################################
class MLPClassifierModel(nn.Module):
    """
    Create a custom PyTorch model that mimics the behavior of a scikit-learn MLPClassifier model
    Define the PyTorch Neural Network model (ReLU MLP-based)
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation=None):
        super(MLPClassifierModel, self).__init__()
        
        activation= activation if activation else nn.ReLU()
        
        self.layers= nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        self.activations= [activation]
        
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.activations.append(activation)
        
        self.output_layer= nn.Linear(hidden_sizes[-1], output_size)

        
    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x= activation(layer(x))
        
        x= self.output_layer(x)
        
        return x


##############################################################
def sklearn_to_pytorch_NN(skl_nn_model, input_size, output_size=1, activation=None):
    """
    convert a scikit-learn NN model to a PyTorch NN model

    skl_nn_model is the scikit-learn Neural Net model
    input_size is the number of input features
    
    RETURNS: a PyTorch Neural Net model used to binary classifications
    binary classification -- one output neuron for the binary prediction
    """
    # mapping from scikit-learn activation string to PyTorch activation function.
    activation_map= {
        'identity': nn.Identity(),
        'logistic': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'relu': nn.ReLU()
    }
    if activation is None:
        activation= activation_map.get(
            getattr(skl_nn_model, 'activation', 'relu'), nn.ReLU()
        )

    if isinstance(skl_nn_model.hidden_layer_sizes, tuple):
        hidden_sizes= list(skl_nn_model.hidden_layer_sizes)
    else:
        [skl_nn_model.hidden_layer_sizes]
    output_size= output_size if output_size else skl_nn_model.n_outputs_
    
    nn_pytorch_model= MLPClassifierModel(input_size, hidden_sizes, output_size, activation)

    # Transfer the weights from the scikit-learn model to the PyTorch model
    for i, layer in enumerate(nn_pytorch_model.layers):
        layer.weight.data= torch.tensor(skl_nn_model.coefs_[i].T, dtype=torch.float32)
        layer.bias.data= torch.tensor(skl_nn_model.intercepts_[i], dtype=torch.float32)

    nn_pytorch_model.output_layer.weight.data= torch.tensor(skl_nn_model.coefs_[-1].T, dtype=torch.float32)
    nn_pytorch_model.output_layer.bias.data= torch.tensor(skl_nn_model.intercepts_[-1], dtype=torch.float32)
    
    return nn_pytorch_model


##############################################################
# Define the XAI explainers to evaluate
# Outputs a dictionary containing feature attribution explanations 
# from various methods as torch.Tensors for each attribution vector.
##############################################################

class XAIExplainers():
    """
    Define the XAI Explainers that will be evaluated
    """
    def __init__(
            self, model, training_data, training_labels, descriptor, cat_fts=[], is_model_NN:bool=False
    ) -> None:
        """
        Initializes the XAIExplainers class.
        Parameters:
        - model: A trained machine learning model (sklearn or XGBoost) -- binary classifier.
        - training_data: DataFrame with the training data.
        - training_labels: DataFrame with the training labels.
        - descriptor: Dictionary defining parameters for the explanation.
        - cat_fts: List of categorical features; if empty, all features are treated as numeric.
        - is_model_NN: Flag indicating whether the model is a neural network.
        """
        self.model= model
        self.data= training_data
        self.labels= training_labels
        
        if (is_model_NN is True):    
            # convert a scikit-learn NN model to a PyTorch NN model used in captum
            # nn_pytorch_model: A PyTorch version of the model used for gradient-based explanation methods.
            self.nn_pytorch_model= sklearn_to_pytorch_NN(model, training_data.shape[1])
        else:
            self.nn_pytorch_model= None

        # retrained: Whether the ohe_model present (for the categorical T-Explainer).
        # ohe_model: A version of model trained on one-hot encoded data (for the categorical T-Explainer).
        self.ohe_model, self.retrained= self.handle_cat_texp(cat_fts, descriptor)
        self.cat_fts= cat_fts


    def handle_cat_texp(self, cat_fts, descriptor):
        """
        Handler for the categorical version of T-Explainer (when cat_fts is not empty)
        RETURNS:
            The categorical-based version of model and the retrained flag
        
        WIP: Import T-Exp
        """
        ohe_model= None
        retrained= False
        """
        if (np.asarray(cat_fts).shape[0]> 0):
            # T-Explainer requires a model retraining only for categorical cases
            self.num_ohe_train_data= ohe_cat_to_numerical_simulator(
                self.data, cat_fts, delta=descriptor['ohe_delta'], rand_seed=True
            )
            ohe_model= clone(self.model)
            ohe_model.fit(self.num_ohe_train_data, self.labels.values.ravel())
            retrained= True
        # """
        return ohe_model, retrained


    def get_x_explanations(self, target_x, target_y, descriptor):
        """
        Generate feature attribution explanations for a single data instance using multiple XAI methods
        - This method can be modified to include/exclude XAI methods to be evaluated by the metrics

        Parameters
        - target_x: DataFrame with the instance to explain.
        - target_y: The true label of the instance to explain.
        RETURNS:
            A dictionary containing feature attribution explanations from various methods as torch.Tensors
                T-Explainer (WIP: Import T-Exp)
                SHAP
                LIME
                Integrated Gradients
                Input X Gradient
                DeepLIFT
                LRP
        """
        assert self.data.shape[1] == target_x.shape[1], "The target data must have the same features of the training data!"
        # ------------------------------------ x_data explanation
        t_x_exp= torch.zeros(self.data.shape[1])
        """
        # ---------- data point explanation -- T-Exp -- WIP
        if (self.retrained):    # Categorical case 
            t_x_exp, t_x_ft, t_x_sh= categorical_taylor_explainer(
                self.ohe_model, self.num_ohe_train_data, self.labels, target_x, target_y, 
                cat_cols=self.cat_fts,
                h_min=descriptor['h_min'], h_max=descriptor['h_max'],
                eps=descriptor['jacobian_eps'], max_itr=descriptor['max_itr'],
                finite_diff_version=descriptor['finite_diff_version'], 
                delta=descriptor['ohe_delta'], e_x=descriptor['e_x'], 
                angle=descriptor['angle'], retrained_ohe_model=True, 
                verbose=False
            )
        else:
            t_x_exp, t_x_ft, t_x_sh= taylor_explainer(
                self.model, self.data, self.labels, target_x, target_y,
                h_min=descriptor['h_min'], h_max=descriptor['h_max'],
                eps=descriptor['jacobian_eps'], max_itr=descriptor['max_itr'],
                finite_diff_version=descriptor['finite_diff_version'], 
                e_x=descriptor['e_x'], angle=descriptor['angle'],
                verbose=False
            )
        t_x_exp= torch.from_numpy(t_x_exp)
        # """

        # ---------- data point explanation -- SHAP Explanation
        if isinstance(self.model, xgb.XGBModel):
            shap_exp_gen= shap.TreeExplainer(self.model, self.data, model_output='probability')
        else:
            shap_exp_gen= shap.Explainer(self.model.predict, self.data)

        shap_x_exp= shap_exp_gen(target_x)
        shap_x_exp= (torch.from_numpy(shap_x_exp.values)).squeeze()


        # ---------- data point explanation -- LIME Explanation
        lime_exp_gen= lime.lime_tabular.LimeTabularExplainer(
            training_data=np.asarray(self.data), 
            feature_names=np.asarray(self.data.columns), 
            training_labels=self.labels.values.ravel().astype(int), 
            class_names=np.asarray([0,1]), 
            mode='classification', 
            discretize_continuous=False, 
            verbose=False
        )

        # ignore temporarily warnings related to feature names
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            lime_scores= lime_exp_gen.explain_instance(
                data_row=np.asarray(target_x)[0], 
                predict_fn=self.model.predict_proba, 
                num_features=self.data.shape[1]
            )

            lime_x_exp= torch.from_numpy(lime_exp_in_data_order(lime_scores, self.data.shape[1]))
        # reset the warning settings
        warnings.resetwarnings()
        
        
        # ---------- data point explanation -- Gradient-based Explanations
        if (self.nn_pytorch_model is not None):
            x_data_tensor= torch.tensor(np.asarray(target_x), dtype=torch.float32, requires_grad=True)
            # ignore non-important warnings temporarily
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Setting")
            
                itGd= IntegratedGradients(self.nn_pytorch_model)
                itGd_x_exp= (itGd.attribute(x_data_tensor)).squeeze().detach()

                iXGd= InputXGradient(self.nn_pytorch_model)
                iXGd_x_exp= (iXGd.attribute(x_data_tensor.unsqueeze(0))).squeeze().detach()

                dLif= DeepLift(self.nn_pytorch_model)
                dLif_x_exp= (dLif.attribute(x_data_tensor.unsqueeze(0))).squeeze().detach()

                lwrp= LRP(self.nn_pytorch_model)
                lwrp_x_exp= (lwrp.attribute(x_data_tensor.unsqueeze(0))).squeeze().detach()
            # reset the warning settings
            warnings.resetwarnings()
        else:
            itGd_x_exp= torch.zeros(self.data.shape[1])
            iXGd_x_exp= torch.zeros(self.data.shape[1])
            dLif_x_exp= torch.zeros(self.data.shape[1])
            lwrp_x_exp= torch.zeros(self.data.shape[1])

        importances= {
            't_exp': t_x_exp, 'lime': lime_x_exp, 'shap': shap_x_exp,
            'itGd': itGd_x_exp, 'iXGd': iXGd_x_exp, 'dLif': dLif_x_exp, 'lwrp': lwrp_x_exp, 
        }
            
        return importances


##############################################################
# classes from OpenXAI (Agarwal, Chirag, et al., 2022)
# we cloned these classes here due to compatibility issues with OpenXAI
# we included in get_perturbed_inputs an optional generator in order to generate
# pseudorandom samplin
##############################################################

class BasePerturbation:
    '''
    Base Class for perturbation methods.
    '''
    
    def __init__(self, data_format):
        '''
        Initialize generic parameters for the perturbation method
        '''
        self.data_format = data_format
    
    def get_perturbed_inputs(self):
        '''
        This function implements the logic of the perturbation methods which will return perturbed samples.
        '''
        pass


class NormalPerturbation(BasePerturbation):
    def __init__(self, data_format, mean: int = 0, std_dev: float = 0.05, flip_percentage: float = 0.3):
        self.mean = mean
        self.std_dev = std_dev
        self.flip_percentage = flip_percentage

        super(NormalPerturbation, self).__init__(data_format)
        '''
        Initializes the marginal perturbation method where each column is sampled from marginal distributions 
        given per variable. dist_per_feature : vector of distribution generators 
        (tdist under torch.distributions).
        Note : These distributions are assumed to have zero mean since they get added to the original sample.
        '''
        pass

    def get_perturbed_inputs(self, original_sample: torch.FloatTensor, feature_mask: torch.BoolTensor,
                             num_samples: int, feature_metadata: list, max_distance: int = None, 
                             generator=None) -> torch.tensor:
        '''
        feature mask : this indicates the static features
        num_samples : number of perturbed samples.
        max_distance : the maximum distance between original sample and purturbed samples.
        generator: generator (torch.Generator, optional) - a pseudorandom number generator for sampling
        '''
        feature_type = feature_metadata
        assert len(feature_mask) == len(
            original_sample), "mask size == original sample in get_perturbed_inputs for {}".format(self.__class__)

        # perturbed_cols = []
        continuous_features = torch.tensor([i == 'c' for i in feature_type])
        discrete_features = torch.tensor([i == 'd' for i in feature_type])

        # Processing continuous columns -- generator inclusion
        mean = self.mean
        std_dev = self.std_dev
        perturbations = torch.normal(
            mean, std_dev, [num_samples, len(feature_type)], generator=generator
        ) * continuous_features + original_sample

        # Processing discrete columns
        flip_percentage = self.flip_percentage
        p = torch.empty(num_samples, len(feature_type)).fill_(flip_percentage)
        perturbations = perturbations * (~discrete_features) + torch.abs(
            (perturbations * discrete_features) - (torch.bernoulli(p) * discrete_features))

        # keeping features static that are in top-K based on feature mask
        perturbed_samples = original_sample * feature_mask + perturbations * (~feature_mask)

        return perturbed_samples


##############################################################
# Quantitative Metrics -- Auxiliar Methods
##############################################################

def pred_proba_to_log_odds(p, eps=1e-9):
    """
    converts probabilities to log odds
    eps is a small positive value to prevent division by zero
    """
    p_clipped= np.clip(p, eps, 1 - eps)
    
    return np.log(p_clipped / (1 - p_clipped))


##############################################################
def ML(model, x, predict_proba=True, lodds:bool=False):
    """
    Computes the predicted probabilities or class labels of a given instance x using a 
    trained machine learning model.

    Parameters:
    - model (object): a trained binary classification model that supports predict_proba- and 
    predict-like methods.
    - x (pd.DataFrame): a data instance (feature vector) for which predictions will be made. If 
    the model was trained with feature names, x should be a DataFrame with matching column names.
    - predict_proba: bool, optional (default=True)
        - If True, returns the predicted probabilities of each class.
        - If False, returns the predicted class labels.
    - lodds: bool, optional (default=False)
        - If True, converts predicted probabilities to log-odds before returning.
        - Only applicable if predict_proba=True.
    RETURNS:
    - np.ndarray
        - If predict_proba=True: Returns an array of predicted probabilities for each class.
        - If predict_proba=False: Returns an array of predicted class labels.
        - If lodds=True: Returns log-odds of predicted probabilities.
    """
    # ignore temporarily warnings related to feature names
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        
        if predict_proba:
            pred= np.squeeze(model.predict_proba(x))  # ensure output shape is 1D
        else:
            pred= model.predict(x)  # predict class labels

        # reset the warning settings
    warnings.resetwarnings()

    if lodds and predict_proba:  # convert predicted probabilities to log-odds
        return pred_proba_to_log_odds(pred)
        
    return pred


##############################################################
def distance_ordering(tensor_x, tensor_y, target_x):
    """
    Order the rows of two torch.Tensor according to their Euclidean distance from a target 
    instance.
    RETURNS: tensor_x and tensor_y (torch.Tensor) ordered by increasing Euclidean distance 
             from target_x
    """
    # calculate Euclidean distances for each row
    distances= torch.norm((tensor_x - target_x), dim=1)
    # sort the data tensor based on distances
    sorted_indices= torch.argsort(distances)
    
    sorted_x= tensor_x[sorted_indices]
    sorted_y= tensor_y[sorted_indices]
    
    return sorted_x, sorted_y


##############################################################
def remove_tensor_row_by_indexset(dataset, index_to_remove):
    """
    Remove a set of rows from a tensor dataset using a set of indices.

    dataset is a tensor dataset with shape (n_rows, ...)
    index_to_remove is a m elements tensor with the indexes to remove
    
    RETURNS: a subset form dataset without the index_to_remove instances
    """
    # ensure indices are unique and sorted (if necessary)
    index_to_remove= torch.unique(index_to_remove)
    
    # generate a mask of rows to keep
    mask= torch.ones(dataset.size(0), dtype=torch.bool)
    mask[index_to_remove]= False  # Set indices to remove as False
    # apply the mask to filter the dataset
    subset= dataset[mask]

    return subset


##############################################################
def get_subsets(x, x_class, dataset, dataset_class, n_elements, option:int=0):
    """
    Obtain a subset from a dataset with at least n_elements.

    x is a tensor instance
    x_class is a tensor with the class of x
    dataset is a m elements tensor dataset
    dataset_class is a m elements tensor with the predicted 
    n_elements is an integer indicating the size of the subset
    
    RETURNS: two tensor subsets (from dataset and dataset_class) with 
             option 1 - n_elements ordered first by class (same from x) and then by distance from x
             option 0 - n_elements ordered by class (same from x) and filled (if necessary) with x and x_class
    
    option 0 gives us y' = y for all x' and option 1 relaxes such a restriction
    """
    data_size= dataset.shape[0]
    
    if (data_size< n_elements):
        raise ValueError("Data size must be greater than n_elements!")

    # Option 1: Order by distance and filter by class
    if option:
        # Order dataset by distance from `x`
        dataset_order, dataset_class_order= distance_ordering(dataset, dataset_class, x.unsqueeze(0))

        # Get indices of elements matching `x_class`
        ind_same_class= (dataset_class_order == x_class).nonzero(as_tuple=True)[0]
        subset= dataset_order[ind_same_class][:n_elements]
        subset_class= dataset_class_order[ind_same_class][:n_elements]

    # Option 0: Filter by class only
    else:
        ind_same_class= (dataset_class == x_class).nonzero(as_tuple=True)[0]
        subset= dataset[ind_same_class][:n_elements]
        subset_class= dataset_class[ind_same_class][:n_elements]

    # If subset size is less than required, fill remaining
    if subset.shape[0] < n_elements:
        remaining= n_elements - subset.shape[0]

        if option:
            # Remove already selected indices and pick the next closest points
            unselected_mask= torch.ones(dataset_order.shape[0], dtype=torch.bool)
            unselected_mask[ind_same_class]= False
            additional= dataset_order[unselected_mask][:remaining]
            additional_class= dataset_class_order[unselected_mask][:remaining]

            subset= torch.cat((subset, additional))
            subset_class= torch.cat((subset_class, additional_class))
        else:
            # Fill with copies of `x` and `x_class`
            fill_x= x.repeat((remaining, 1))
            fill_x_class= x_class.repeat(remaining)

            subset= torch.cat((subset, fill_x))
            subset_class= torch.cat((subset_class, fill_x_class))

    return subset, subset_class


##############################################################
def clip_small_values(v, eps=1e-6):
    """
    clip values near to zero in v replacing them with eps

    - v is a single value (float) or a numpy.ndarray with (n,) shape
    - eps is a small number of tolerance limiting what is a small value

    RETURNS: v with values whose absolute value is below eps replaced
             with eps (preserving the sign).
    """
    if isinstance(v, np.ndarray):
        # vectorized clipping for arrays
        v_clipped= np.where((v < 0) & (np.abs(v) < eps), -eps, v)
        v_clipped= np.where((v > 0) & (v < eps), eps, v_clipped)
    else:
        # scalar clipping
        if (v < 0 and abs(v) < eps):
            v_clipped = -eps
        elif (v > 0 and v < eps):
            v_clipped = eps
        else:
            v_clipped = v

    return v_clipped


##############################################################
def square_difference(v1, v2):
    """
    Compute the element-wise square of the difference between two arrays.
    RETURNS: the square of the difference of the corresponding elements 
             in v1 and v2.
    """
    # arrays can be flattened, so long as ordering is preserved
    v1_flat= np.asarray(v1).flatten()
    v2_flat= np.asarray(v2).flatten()
    
    dif_flat= (v1_flat - v2_flat)
    
    return np.power(dif_flat, 2)


##############################################################
def lp_norm_dif(v1, v2, p_norm=2, eps=1e-6, norm:bool=True):
    """
    Normalizes the difference between v1 and v2 by v1 (adapted; Agarwal, 
    Chirag, et al., 2022)
    RETURNS: the Lp norm of the difference between v1 and v2.
    """
    # arrays can be flattened, so long as ordering is preserved
    v1_flat= np.asarray(v1).flatten()
    v2_flat= np.asarray(v2).flatten()
    
    dif_flat= (v1_flat - v2_flat)
    
    if (norm is True):
        v1_flat= clip_small_values(v1_flat, eps)
        
        dif_flat= np.divide(dif_flat, v1_flat, out=np.zeros_like(v1_flat), where=v1_flat!=0)

    return np.linalg.norm(dif_flat, ord=p_norm)


##############################################################
def ris_measure(x_data, x_pert, exp_data, exp_pert, p_norm=2, eps=1e-6):
    """ 
    compute norm between predictions per perturbation - RIS 
    """
    x_dif_norm= lp_norm_dif(x_data, x_pert, p_norm=p_norm, eps=eps, norm=True)
    # x_dif_norm= np.clip(x_dif_norm, eps, None)
    x_dif_norm= clip_small_values(x_dif_norm, eps)
    
    exp_dif_norm= lp_norm_dif(exp_data, exp_pert, p_norm=p_norm, eps=eps, norm=True)
    
    stability_measure= np.divide(exp_dif_norm, x_dif_norm, where=x_dif_norm!=0)
    
    return stability_measure


##############################################################
def ros_measure(fx_data, fx_pert, exp_data, exp_pert, p_norm=2, eps=1e-6):
    """
    compute norm between representations - ROS
    x_data and x_pert must to be pd.DataFrame row individual instances with column names
    """
    fx_dif_norm= lp_norm_dif(fx_data, fx_pert, p_norm=p_norm, eps=eps, norm=True)
    # fx_dif_norm= np.clip(fx_dif_norm, eps, None)
    fx_dif_norm= clip_small_values(fx_dif_norm, eps)
    
    exp_dif_norm= lp_norm_dif(exp_data, exp_pert, p_norm=p_norm, eps=eps, norm=True)

    stability_measure= np.divide(exp_dif_norm, fx_dif_norm, where=fx_dif_norm!=0)
    
    return stability_measure


##############################################################
def lime_exp_in_data_order(lime_exp, num_fts):
    """
    bring explanations into data order (since LIME automatically orders according 
    to highest importance)
    """
    exp= np.zeros(num_fts)

    for k, v in lime_exp.local_exp[1]:
        exp[k]= v

    return exp


##############################################################
# Metric -- Relative Input/Output Stability -- RIS / ROS
##############################################################

def relative_stability(model, explainers, data, labels, perturbation, descriptor, generator=None):
    """
    Relative Input/Output Stability (RIS/ROS) Metric Computation
    This function evaluates the stability of feature attribution explanations 
    by computing the Relative Input Stability (RIS) and Relative Output Stability (ROS) metrics.
    - Flexible to evaluate any XAI method in get_x_explanations

    Parameters
    - model: A trained machine learning model (sklearn or XGBoost) -- binary classifier.
    - explainers: XAIExplainers object used to define the XAI methods to evaluate
    - data: DataFrame with data used to evaluate the XAI methods.
    - labels: DataFrame with data's labels used to evaluate the XAI methods.
    - perturbation (OpenXAI object): Object used to generate perturbed samples.
    - descriptor: Dictionary defining parameters.
    - generator (torch.Generator, optional): Pseudorandom number generator for sampling.
    RETURNS:
        dict: Dictionary containing overall RIS and ROS metrics for each explanation method
    """
    exp_methods = None  # will be set on the first instance
    method_ris_max_ratios = {}
    method_ris_mean_ratios= {}
    method_ros_max_ratios = {}
    method_ros_mean_ratios= {}
    
    data_size= data.shape[0]
    
    for i_data in tqdm(range(data_size)):
        # i_data and its label as pd.DataFrames
        target_x= pd.DataFrame(data=[data.iloc[i_data,:]], columns=data.columns)
        target_y= pd.DataFrame(data=[labels.iloc[i_data]], columns=labels.columns)

        # i_data as a tensor
        x_data= torch.tensor(target_x.values, dtype=torch.float64)
        # data point prediction
        y_pred= torch.from_numpy(ML(model, target_x, predict_proba=False).astype(int))
        fx_data= ML(model, target_x)


        # ------------------------------------ get x_data explanation
        x_exps= explainers.get_x_explanations(target_x, target_y, descriptor)
        
        if exp_methods is None:
            # first procedure only for the first iteraction on data (same methods during the entire processing)
            exp_methods = list(x_exps.keys())  # get the methods under evaluation
            # create dictionaries to store RIS/ROS results for each method over multiple runs
            method_ris_max_ratios = {method: [] for method in exp_methods}
            method_ris_mean_ratios= {method: [] for method in exp_methods}
            method_ros_max_ratios = {method: [] for method in exp_methods}
            method_ros_mean_ratios= {method: [] for method in exp_methods}

        
        # ------------------------------------ x_data perturbation
        # data point perturbation
        x_pert_samples= perturbation.get_perturbed_inputs(
            original_sample=x_data.reshape(-1), 
            feature_mask=descriptor['mask'],
            num_samples=descriptor['num_samples'], 
            max_distance=descriptor['pert_max_distance'],
            feature_metadata=descriptor['feature_metadata'], 
            generator=generator
        )

        # --- take the closest num_perts points to x_data that have the same predicted class label to x_data
        y_pert_preds= torch.from_numpy(ML(
            model, pd.DataFrame(data=x_pert_samples.numpy(), columns=data.columns), predict_proba=False
        ).astype(int))
        
        # get only the first num_perts points ordered by class and distance from x_data
        x_pert_samples, y_pert_preds= get_subsets(
            x_data.reshape(-1), y_pred, x_pert_samples, y_pert_preds, descriptor['num_perts']
        )
        
        
        # ------------------------------------ explain each x_data perturbation
        method_exp_pert_samples= {
            method: torch.zeros_like(x_pert_samples) for method in exp_methods
        }
        method_x_ris_ratios= {method: [] for method in exp_methods}
        method_x_ros_ratios= {method: [] for method in exp_methods}
        
        # for each perturbation sample, compute the explanation and its stability
        for i, x_pert in enumerate(x_pert_samples):
            
            df_x_pert= pd.DataFrame(data=[x_pert.numpy()], columns=data.columns)
            df_y_pert= pd.DataFrame(data=[np.int64(y_pert_preds[i])], columns=labels.columns)
            fx_pert= ML(model, df_x_pert)
            
            # ------------------------------------ x_pert explanation
            x_exp_pert= explainers.get_x_explanations(df_x_pert, df_y_pert, descriptor)
            
            for method in exp_methods:
                # store the explanation for the i-th perturbation
                method_exp_pert_samples[method][i, :]= x_exp_pert[method]
        
    
            # ------------------------------------ get stability for each explanation method
            # RIS / ROS --- one processing cicle
            for method in exp_methods:
                ris_val= ris_measure(
                    x_data, x_pert, x_exps[method], method_exp_pert_samples[method][i], 
                    p_norm=descriptor['p_norm'], eps=descriptor['eps_norm']
                )
                ros_val= ros_measure(
                    fx_data, fx_pert, x_exps[method], method_exp_pert_samples[method][i], 
                    p_norm=descriptor['p_norm'], eps=descriptor['eps_norm']
                )
                method_x_ris_ratios[method].append(ris_val)
                method_x_ros_ratios[method].append(ros_val)
        
        # --- for the current instance, record the max and mean stability values per method
        for method in exp_methods:
            method_ris_max_ratios[method].append(max(method_x_ris_ratios[method]))
            method_ris_mean_ratios[method].append(np.mean(method_x_ris_ratios[method]))
            
            method_ros_max_ratios[method].append(max(method_x_ros_ratios[method]))
            method_ros_mean_ratios[method].append(np.mean(method_x_ros_ratios[method]))

           
    # ------------------------------------ RETURN ratios considering all data processed
    results = {}
    for method in exp_methods:
        ris_max_arr = np.asarray(method_ris_max_ratios[method])
        ris_mean_arr= np.asarray(method_ris_mean_ratios[method])
        ros_max_arr = np.asarray(method_ros_max_ratios[method])
        ros_mean_arr= np.asarray(method_ros_mean_ratios[method])
        
        results[f'{method}_ris_max'] = float(ris_max_arr.max())
        results[f'std({method}_ris_max)'] = float(ris_max_arr.std())
        results[f'{method}_ris_mean'] = float(ris_mean_arr.mean())
        results[f'std({method}_ris_mean)'] = float(ris_mean_arr.std())
        results[f'{method}_ros_max'] = float(ros_max_arr.max())
        results[f'std({method}_ros_max)'] = float(ros_max_arr.std())
        results[f'{method}_ros_mean'] = float(ros_mean_arr.mean())
        results[f'std({method}_ros_mean)'] = float(ros_mean_arr.std())

    # the max/mean stability ratios
    return results


##############################################################
# Metric -- Run Explanation Stability -- RES
##############################################################

def run_stability(explainers, data, labels, descriptor):
    """
    Computes explanation stability over multiple runs, returning the highest instability for each method.
    - Flexible to evaluate any XAI method in get_x_explanations

    Parameters
    - explainers: XAIExplainers object used to define the XAI methods to evaluate
    - data: DataFrame with data used to evaluate the XAI methods.
    - labels: DataFrame with data's labels used to evaluate the XAI methods.
    - descriptor: Dictionary defining parameters.
    RETURNS:
        Dictionary of max instability values for each explanation method.
    """
    exp_methods = None  # will be set on the first instance
    stability_ratios: dict[str, list[float]]= {}
    
    # number of runs for stability computation
    runs= int(descriptor['num_runs'])
    data_size= data.shape[0]
    
    for i_data in tqdm(range(data_size)):
        # x_data and its label as pd.DataFrame
        target_x= pd.DataFrame(data=[data.iloc[i_data,:]], columns=data.columns)
        target_y= pd.DataFrame(data=[labels.iloc[i_data]], columns=labels.columns)
        
        method_x_exps= {}
        
        for i in range(runs):
            # ------------------------------------ n runs x_data explanation
            x_exps= explainers.get_x_explanations(target_x, target_y, descriptor)
            
            if exp_methods is None:
                # first procedure only for the first iteraction of this loop
                exp_methods = list(x_exps.keys())  # get the methods under evaluation
            
            if i==0:
                # create a dictionary to store explanations for each method over multiple runs
                method_x_exps= {method: [] for method in exp_methods}

            # get the explanation of each method to each run
            for method in exp_methods:
                method_x_exps[method].append(x_exps[method].numpy())  # convert to numpy


        # convert to numpy arrays
        for method in exp_methods:
            method_x_exps[method]= np.asarray(method_x_exps[method])

        # compute mean explanation for each method
        method_x_exps_mean= {method: np.mean(x_exps, axis=0) for method, x_exps in method_x_exps.items()}
        
        
        method_x_exp_ratios= {method: [] for method in exp_methods}
        # ------------------------------------ distance of each explanation from the mean of explanations
        method_x_exp_ratios= {
            method: [
                lp_norm_dif(method_x_exps_mean[method], x_exps[j], p_norm=descriptor['p_norm'], norm=False) 
                for j in range(runs)
            ] for method, x_exps in method_x_exps.items()
        }

        if i_data== 0:
            # only for the first iteraction on the data
            stability_ratios= {method: [] for method in exp_methods}
        
        # ------------------------------------ max ratio related to each x_data
        for method, x_exp_ratios in method_x_exp_ratios.items():
            stability_ratios[method].append(x_exp_ratios[np.argmax(x_exp_ratios)])

    
    # ------------------------------------ overall max ratio for each method related to all data
    results = {method: float(max(ratios)) for method, ratios in stability_ratios.items()}    

    # max stability_ratios
    return results


##############################################################
# Metric -- Prediction Gap on Important Features -- PGI
##############################################################

def sorted_indices(seq, reverse:bool=False):
    """
    RETURNS: the indexes of a descending sorted array (seq) if reverse is True, indexes of an 
    ascending sorted array if reverse is False
    """
    seq= np.asarray(seq)
    
    if reverse: seq= -seq
    
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]


##############################################################
def get_top_k_x_noise(x, e_index, top_k, noise_type='zero', perturbation=None):
    """
    x is a Pandas DataFrame with an instance
    e_index is a vector of indices from an explanation ordered with sorted_indices()
    top_k is an INTEGER representing the number of top features
    x_pert represents the zero/perturbed instance from x used to generate a x'
    
    TODO: implement different types of noise
    
    RETURNS: a perturbed instance xi' [i is in top_k]
    """
    # get the top k features
    top_k_index= e_index[:top_k]
    names= x.columns

    top_k_names= [names[v] for (i, v) in enumerate(top_k_index)]
    # non_top_k_names= list(filter(lambda x:x not in top_k_names, names))

    x_pert= np.zeros(x.shape)
    x_pert_sample= pd.DataFrame(data=x_pert, columns=x.columns)

    # delete/perturb the top-k important features to produce x'
    x_noise= (x.copy()).reset_index(drop=True)
    x_noise[top_k_names]= x_pert_sample[top_k_names]

    return x_noise


##############################################################
def eval_pred_faithfulness(model, explainers, data, labels, descriptor, top_k=1, noise_type='zero', perturbation=None):
    """
    Prediction Gap on Important Features
    - model: A trained machine learning model (sklearn or XGBoost) -- binary classifier.
    - explainers: XAIExplainers object used to define the XAI methods to evaluate
    - data: DataFrame with data used to evaluate the XAI methods.
    - labels: DataFrame with data's labels used to evaluate the XAI methods.
    - descriptor define the parameters to explanations and data perturbations
    - top_k (int): Number of important features to perturb (can be a list).
    - noise_type: Type of noise to use for feature perturbation. Default is 'zero'
    - perturbation is a OpenXAI perturbation object
    
    RETURNS: PGIF metric considering all features as important for T-Exp, SHAP, and LIME.
    
    The PGIF scores provide insights into the model's predictions, considering both the change in accuracy 
    when the feature is randomized and the difference in Exp values.
    """
    # ensure everything is integer, upper and lower boundaries, and remove repetitions
    top_k = sorted(set(max(1, min(int(x), data.shape[1])) for x in np.reshape([top_k], -1)))

    exp_methods= None
    methods_pgi= None
    
    data_size= data.shape[0]
    
    for i_data in tqdm(range(data_size)):
        # i_data and its label as pd.DataFrames
        target_x= pd.DataFrame(data=[data.iloc[i_data,:]], columns=data.columns)
        target_y= pd.DataFrame(data=[labels.iloc[i_data]], columns=labels.columns)
        
        # get the predicted probability for f(x)
        true_label_value= int(target_y.values.item())
        fx_acc= ML(model, target_x)[true_label_value]
        
        
        # ------------------------------------ get x_data explanation
        x_exps= explainers.get_x_explanations(target_x, target_y, descriptor)
        
        if exp_methods is None:
            # first procedure only for the first iteraction on data (same methods during the entire processing)
            exp_methods = list(x_exps.keys())  # get the methods under evaluation
            # create a dictionary to store PGI results for each method for each top_k value
            methods_pgi = {method: [[] for _ in top_k] for method in exp_methods}


        # ------------------------------------ identify the indices of the top-k important features
        """ Magnitude Indicates Importance: The absolute value of the importance score usually reflects 
        the feature's importance. Large negative or positive values both indicate high importance; the 
        sign simply tells you the direction of the feature's contribution to the prediction (positive 
        or negative impact).
        """
        # create a dictionary to store explanations' sorted indices for each method
        method_importances_index= {
            method: sorted_indices(np.abs(x_exps[method].numpy()), reverse=True)
            for method in exp_methods
        }
        
        # ------------------------------------ iterate for each top_k value of each explanation
        for j, top_k_value in enumerate(top_k):
            # delete/perturb features that are in the top k ones to produce x'
            method_target_x = {
                method: get_top_k_x_noise(
                    target_x, method_importances_index[method], top_k_value, noise_type, perturbation
                )
                for method in exp_methods
            }
        
            # take the difference between f(x) and f(x')
            method_fx_acc = {
                method: ML(model, method_target_x[method])[true_label_value]
                for method in exp_methods
            }

            # take the difference between f(x) and f(x')
            method_fx_acc_diff = {
                method: np.abs(fx_acc - method_fx_acc[method])
                for method in exp_methods
            }
            
            # compute the mean (1/m)sum(|f(x) - f(x'm)|)
            # high fidelity explanation will result in high accuracy differences when deleting/perturbing
            # the k most important features
            for method in exp_methods:
                methods_pgi[method][j].append(np.mean(method_fx_acc_diff[method]))


    results= {'PGI values from top k features': top_k}
    results.update({
        f'{method}_pgi': np.mean(values, axis=1).tolist() 
        for method, values in methods_pgi.items()
    })
    results.update({
        f'{method}_pgi_std': np.std(values, axis=1).tolist() 
        for method, values in methods_pgi.items()
    })

    return results


##############################################################
# Metric -- Local Accuracy Preservation -- LAP
##############################################################

def expected_value_x_mean(model, x_mean, y_train, lodds:bool=False):
    """
    Expected value can be understood as the average model output across the training set, and the true labels
    SHAP paper mention a dataset "would be predicted if we did not know any features"

    The 'absence of features' or better 'not knowing the feature' needs to be defined/considered carefully. 
    In the context of SHAP it doesn't meant that Xi=0 but it means that we do not know the value of Xi
    but we still may know the distribution of potential values of Xi or we could estimate this distribution 
    based on the data, in practice, a mean value is considered, and then, we average over this predicted 
    probs to each label.

    Receive one instance x_mean from train dataset mean and y_train, the entire labels_train

    IF lodds true return a probability values, log-odds otherwise 

    RETURNS: the mean probability of x_mean belonging to the in y_train classes
    """
    # Expected value over the mean from train data
    p_x= ML(model, x_mean)

    tam= len(y_train) if len(y_train)<= 10000 else 10000
    pred= 0
    
    for i in range(tam):
        pred= pred+ p_x[y_train[i].astype(int)]
        
    avg_pred= pred/ tam

    if lodds:
        return float(pred_proba_to_log_odds(avg_pred))
    
    return float(avg_pred)


##############################################################
def eval_local_accuracy(model, explainers, data, labels, descriptor, train_data, labels_train, 
                        phi_0_methods_list= ['t_exp', 'lime'],
                        shap_methods_list = ['shap'],
                        grad_methods_list = ['itGd', 'iXGd', 'dLif', 'lwrp']):
    """
    For an additive explanator, local accuracy preservation means that the sum of all feature importance values
    will be equal to the difference between the expected value of the model and the predicted value, i.e.,
    f(x) = phi_0 + Sum(phi_i), with phi_0 = E[f(X)]. This function computes a local accuracy preservation 
    (LAP) metric for each explanation method as the fraction of instances for which the additive approximation.
    - Flexible to evaluate any XAI method in get_x_explanations. If necessary, edit the lists of methods

    Parameters:
    - model: A trained machine learning model (sklearn or XGBoost) -- binary classifier.
    - explainers: XAIExplainers object used to define the XAI methods to evaluate
    - data: DataFrame with data used to evaluate the XAI methods.
    - labels: DataFrame with data's labels used to evaluate the XAI methods.
    - descriptor define the parameters to explanations and data perturbations
    - train_data, labels_train: DataFrames with model's training data and labels
    - phi_0_methods_list is a list with XAI methods under evaluation that use phi_0 as base values
    - shap_methods_list is a list with the versions of SHAP under evaluation (use SHAP base values)
    - grad_methods_list is a list with the gradient-based XAI methods under evaluation (use phi_0_logit)

    RETURNS: measured of faithfulness (dictionary) for local accuracy preservation for each explainer over 
             a non-perturbed dataset as a ratio of explanations that preserved local accuracy. Higher values 
             indicate that more instances preserved local accuracy (i.e., the explanation faithfully 
             decomposes the prediction).
    """
    # we assume all data are numerical and compute the mean of training data to approximate E[f(X)]
    mean_inst= dp.replace_values(
        train_data, train_data.columns.tolist(), num_type='mean', cat_type='none'
    )
    # e_fx can be understood as the average model output across the training set X when Xi is not known
    # for this reason, the data mean is used
    phi_0= expected_value_x_mean(model, mean_inst, labels_train.values.ravel())
    phi_0_logit= pred_proba_to_log_odds(phi_0)

    # compute SHAP base value (for the first instance; assumed as constant)
    if isinstance(model, xgb.XGBModel):
        shap_exp_gen= shap.TreeExplainer(model, train_data, model_output='probability')
    else:
        shap_exp_gen= shap.Explainer(model.predict, train_data)

    shap_x_exp= shap_exp_gen(pd.DataFrame(data=[data.iloc[0,:]], columns=data.columns))
    shap_phi_0= (shap_x_exp.base_values)[0]
    
    # containers to register the XAI methods and accumulate counts for each method
    exp_methods= None
    methods_lap= None
    
    tolerance= descriptor['eps_eval_add']
    data_size= data.shape[0]
    
    for i_data in tqdm(range(data_size)):
        # i_data and its label as pd.DataFrames
        target_x= pd.DataFrame(data=[data.iloc[i_data,:]], columns=data.columns)
        target_y= pd.DataFrame(data=[labels.iloc[i_data]], columns=labels.columns)
        
        # get the predicted probability of f(x)
        fx_p_class=(ML(model, target_x, predict_proba=False).astype(int))[0]
        fx_p_prob = ML(model, target_x)[fx_p_class]
        
        fx_tol_pls= fx_p_prob + tolerance
        fx_tol_min= fx_p_prob - tolerance
        
        logit_fx_tol_pls= pred_proba_to_log_odds(fx_tol_pls)
        logit_fx_tol_min= pred_proba_to_log_odds(fx_tol_min)
    
        # ------------------------------------ get x_data explanation        
        x_exps= explainers.get_x_explanations(target_x, target_y, descriptor)
        
        if exp_methods is None:
            # first procedure only for the first iteraction on data (same methods during the entire processing)
            exp_methods = list(x_exps.keys())  # get the methods under evaluation
            # create a dictionary to store LAP results for each method
            methods_lap = {method: 0 for method in exp_methods}
        

        # ------------------------------------ get the additive approximation for each method
        # convert explanation tensor to numpy array, sum its elements, add the appropriate base value
        method_fx= {method: np.sum(x_exps[method].numpy()) for method in exp_methods}
        
        for method in exp_methods:
            if method in phi_0_methods_list:
                method_fx[method] += phi_0        # XAI methods that use phi_0
            elif method in shap_methods_list:
                method_fx[method] += shap_phi_0   # SHAP phi_0
            else:
                method_fx[method] += phi_0_logit  # Gradient-based methods use phi_0_logit

            # check if the approximation is within tolerance and increment counters for 
            # methods that preserve local accuracy
            if method in grad_methods_list:
                if (method_fx[method]>= logit_fx_tol_min and method_fx[method]<= logit_fx_tol_pls):
                    methods_lap[method] += 1
            else:
                if (method_fx[method]>= fx_tol_min and method_fx[method]<= fx_tol_pls):
                    methods_lap[method] += 1
            

    # compute the fraction of instances preserving local accuracy for each method.
    results = {f"{method} LAP": methods_lap[method] / data_size for method in exp_methods}
    
    return results


##############################################################
# Metric -- Consistency Across Instances -- CAI
##############################################################

def explanation_consistency(explainers, train_data, labels_train, target_x, target_y, descriptor, k=5):
    """
    Compute the average cosine similarity between the explanation of a target instance and the 
    explanations of its k nearest neighbors (restricted to those with the same class label).
    Evaluate how similar the explanations are for similar instances. For example, if two instances 
    are close in feature space, their explanations should also be similar.
    - Flexible to evaluate any XAI method in get_x_explanations

    Parameters
    - explainers: XAIExplainers object used to define the XAI methods to evaluate
    - train_data (pd.DataFrame): The training data, where each row is an instance.
    - labels_train (pd.DataFrame): The training labels as a single-column DataFrame.
    - target_x (pd.DataFrame): A single-row DataFrame representing the target instance.
    - target_y (pd.DataFrame): A single-row DataFrame (with one column) containing the target's label.
    - descriptor: Dictionary defining parameters.
    - k (int, optional): The number of nearest neighbors to consider (default is 5).

    RETURNS:
    - float: The average cosine similarity between the explanation of the target instance and those of 
    its k nearest neighbors.
        - Higher values (closer to 1) indicate that similar instances receive similar explanations.
        - Lower values suggest that the explanations vary more among similar instances.
    """
    # assume labels_train has one column; get its name and extract labels
    label_col = labels_train.columns[0]
    target_label = target_y.iloc[0, 0]  # get the target's label (assumes one value)
    
    # filter train_data to keep only instances with the same label as the target
    train_data_same = train_data[labels_train[label_col] == target_label]
    
    # convert train_data_same and target_x to numpy arrays
    train_data_same_np = train_data_same.values  # shape: (n_instances, n_features)
    target_x_np = target_x.values  # shape: (1, n_features)
    
    # fit NearestNeighbors on the filtered training data
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(train_data_same_np)
    distances, indices = nn.kneighbors(target_x_np)


    # ------------------------------------ get target_x explanation
    x_exps= explainers.get_x_explanations(target_x, target_y, descriptor)
    exp_methods = list(x_exps.keys())
    # convert each explanation tensor to a 2D numpy array
    x_exps= {method: x_exps[method].reshape(1, -1).numpy() for method in exp_methods}
    # dictionary to store neighbor explanations for each method
    neighbor_exps_dict = {method: [] for method in exp_methods}
    
    for idx in indices[0]:
        # retrieve neighbor instances (as a DataFrame)
        X_neighbor = train_data_same.iloc[[idx]]

        # ------------------------------------ get X_neighbor explanation
        neighbor_exps = explainers.get_x_explanations(X_neighbor, target_y, descriptor)

        for method in exp_methods:
            neighbor_exps_dict[method].append(neighbor_exps[method].numpy())  # convert to numpy

    results = {}
    # compute cosine similarities (target explanation vs. each neighbor's explanation)
    for method in exp_methods:
        # convert neighbor explanations to 2D array (k, n_features)
        neighbor_exps = np.asarray(neighbor_exps_dict[method])
        similarity = cosine_similarity(x_exps[method], neighbor_exps)

        # for a single target, cosine_similarity returns a 1 x k vector.
        results[f"{method}_cai"] = np.mean(similarity)

    return results


##############################################################
def mean_explanation_consistency(explainers, train_data, labels_train, descriptor, k=5):
    """
    Compute the mean explanation consistency across all instances in the training data.
    For each instance in the training data (used as the target), this function calls 
    the explanation_consistency method to compute the cosine similarity between the target's explanation 
    and its k nearest neighbors (restricted to those with the same class label). It then returns the 
    average consistency value for each explainer.
    """
    consistency_sums = {}
    count = 0

    # iterate over every instance in train_data
    for idx in tqdm(range(len(train_data))):
        # use the current instance as the target
        x_target = train_data.iloc[[idx]]    # single-row DataFrame
        y_target = labels_train.iloc[[idx]]  # single-row DataFrame
        
        # compute explanation consistency for the target instance
        results = explanation_consistency(
            explainers, train_data, labels_train, x_target, y_target, descriptor, k
        )
        
        # sum up the consistency values for each explainer metric
        for key, value in results.items():
            consistency_sums[key] = consistency_sums.get(key, 0) + value
        count += 1

    # compute the mean consistency for each explainer metric
    mean_consistency = {key: value / count for key, value in consistency_sums.items()}
    
    return mean_consistency
