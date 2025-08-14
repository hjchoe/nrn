import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nrn.utils.mic import compute_mic_matrix

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from nrn.models import BanditNRNClassifier
from nrn.utils.trainers import BanditNRNTrainer

from ..datasets.simple_dataset import SimpleDataset
from ..base.base_estimator import BaseSKLogicEstimator


class RNRNClassifier(BaseSKLogicEstimator):

    """
    Scikit-learn–compatible Neuro‑Reasoning classifier.

    This estimator wraps a Bandit-driven Neuro‑Reasoning Network (NRN) that learns
    sparse logical structures over (optionally) binarized features and exposes a
    familiar `fit/predict` API together with local, textual explanations via
    `explain_sample`.

    Notes
    -----
    - Supports binary and multi-label classification. Set ``multi_class=True`` to
      evaluate with multi-class metrics during training; local explanations are
      currently unavailable for multi-class.
    - The model expects pandas DataFrames at inference/training time. Numpy arrays
      are accepted and will be converted internally.
    """

    def __init__(
            self,
            target_names: list = None,
            feature_names: list = None,
            layer_sizes: list = [8, 8],
            n_selected_features_input: int = 4,
            n_selected_features_internal: int = 4,
            n_selected_features_output: int = 4,
            perform_prune_quantile: float = 0.5,
            ucb_scale: float = 1.5,
            prune_strategy: str = 'class',
            delta: float = 2.0,
            bootstrap: bool = False,
            swa: bool = False,
            add_negations: bool = False,
            normal_form: str = 'cnf',
            weight_init: float = 0.2,
            logits: bool = True,
            # binarization: bool = True,
            binarization: bool = False,
            tree_num: int = 10,
            tree_depth: int = 5,
            tree_feature_selection: float = 0.5,
            thresh_round: int = 3,
            loss_func=nn.BCEWithLogitsLoss,
            learning_rate: float = 0.1,
            weight_decay: float = 0.001,
            t_0: int = 3,
            t_mult: int = 2,
            epochs: int = 200,
            batch_size: int = 32,
            holdout_pct: float = 0.2,
            early_stopping_plateau_count: int = 20,
            perform_prune_plateau_count: int = 3,
            increase_prune_plateau_count: int = 10,
            increase_prune_plateau_count_plateau_count: int = 10,
            lookahead_steps: int = 0,
            lookahead_steps_size: float = 0.0,
            evaluation_metric=roc_auc_score,
            multi_class: bool = False,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            num_workers: bool = False
    ):
        """
        Initialize an RNRNClassifier with architecture, pruning, and training options.

        Parameters
        ----------
        target_names : list of str, optional
            Human‑readable class names. If ``None``, names like ``"Class 0"`` are
            generated from the target columns during ``fit``.
        feature_names : list of str, optional
            Feature names to use when input is not a DataFrame. When ``X`` is a
            :class:`pandas.DataFrame`, its columns are used instead.
        layer_sizes : list of int, default=[8, 8]
            Hidden layer widths of the reasoning network.
        n_selected_features_input : int, default=4
            Number of features sampled per input logic unit.
        n_selected_features_internal : int, default=4
            Number of features sampled per hidden logic unit.
        n_selected_features_output : int, default=4
            Number of hidden features sampled per output logic unit.
        perform_prune_quantile : float, default=0.5
            Quantile of logic to prune during the pruning phase.
        ucb_scale : float, default=1.5
            Exploration scale for the bandit policy (Upper-Confidence Bound).
        prune_strategy : {'class', 'logic', 'class_logic'}, default='class'
            Strategy used when pruning logic units.
        delta : float, default=2.0
            Factor to reduce the likelihood of sampling recently pruned logic during growth.
        bootstrap : bool, default=False
            If ``True`` and a logic‑level prune strategy is used, evaluate with bootstrap
            sampling when pruning.
        swa : bool, default=False
            If ``True``, use Stochastic Weight Averaging during training.
        add_negations : bool, default=False
            If ``True``, include negated logic at initialization.
        normal_form : {'cnf', 'dnf'}, default='cnf'
            Logical normal form used by the network.
        weight_init : float, default=0.2
            Initialization magnitude for logic weights.
        logits : bool, default=True
            If ``True``, model outputs are logits; otherwise they are unscaled probabilities.
        binarization : bool, default=False
            If ``True``, enable tree‑based feature binarization prior to learning.
        tree_num : int, default=10
            Number of trees used for feature binarization.
        tree_depth : int, default=5
            Depth of the binarization trees.
        tree_feature_selection : float, default=0.5
            Fraction of features sampled per split during binarization.
        thresh_round : int, default=3
            Decimal rounding applied to thresholds from tree binarization.
        loss_func : callable, default=torch.nn.BCEWithLogitsLoss
            PyTorch loss constructor. It will be instantiated with no args.
        learning_rate : float, default=0.1
            Learning rate for the AdamW optimizer.
        weight_decay : float, default=0.001
            L2 weight decay for AdamW.
        t_0 : int, default=3
            Initial period for :class:`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.
        t_mult : int, default=2
            Multiplicative factor for the restart period.
        epochs : int, default=200
            Number of training epochs.
        batch_size : int, default=32
            Batch size for DataLoaders.
        holdout_pct : float, default=0.2
            Fraction of the training set reserved as validation for early stopping.
        early_stopping_plateau_count : int, default=20
            Stop training if the validation metric does not improve for this many epochs.
        perform_prune_plateau_count : int, default=3
            Trigger a pruning phase after this many stagnant epochs.
        increase_prune_plateau_count : int, default=10
            Amount by which to increase ``perform_prune_plateau_count`` after pruning.
        increase_prune_plateau_count_plateau_count : int, default=10
            Additional increment applied when repeated plateaus are observed.
        lookahead_steps : int, default=0
            Number of lookahead optimization steps (0 disables lookahead).
        lookahead_steps_size : float, default=0.0
            Step size for lookahead optimization.
        evaluation_metric : callable, default=sklearn.metrics.roc_auc_score
            Metric function used to select the best checkpoint during training.
        multi_class : bool, default=False
            If ``True``, treat the problem as multi‑class during evaluation.
        pin_memory : bool, default=False
            Passed to PyTorch DataLoader for faster host‑to‑GPU transfers.
        persistent_workers : bool, default=False
            Keep DataLoader workers alive between iterations (PyTorch 1.7+).
        num_workers : int, default=0
            Number of DataLoader worker processes. (Note: the constructor type hints
            declare ``bool`` but the intended type is ``int``.)

        Notes
        -----
        This constructor only stores hyperparameters. The underlying model and trainer
        are created during `fit`.
        """
        super(RNRNClassifier, self).__init__(
            binarization=binarization,
            tree_num=tree_num,
            tree_depth=tree_depth,
            tree_feature_selection=tree_feature_selection,
            thresh_round=thresh_round,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            t_0=t_0,
            t_mult=t_mult,
            epochs=epochs,
            batch_size=batch_size,
            holdout_pct=holdout_pct,
            early_stopping_plateau_count=early_stopping_plateau_count,
            lookahead_steps=lookahead_steps,
            lookahead_steps_size=lookahead_steps_size,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_workers=num_workers
        )
        # handle empty target names
        if target_names:
            self.target_names = target_names
        else:
            self.target_names = None

        # handle empty feature names
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = None

        # add hyper-parameters
        self.layer_sizes = layer_sizes
        self.n_selected_features_input = n_selected_features_input
        self.n_selected_features_internal = n_selected_features_internal
        self.n_selected_features_output = n_selected_features_output
        self.perform_prune_quantile = perform_prune_quantile
        self.ucb_scale = ucb_scale
        self.prune_strategy = prune_strategy
        self.delta = delta
        self.bootstrap = bootstrap
        self.swa = swa
        self.add_negations = add_negations
        self.normal_form = normal_form
        self.weight_init = weight_init
        self.logits = logits
        self.loss_func = loss_func
        self.perform_prune_plateau_count = perform_prune_plateau_count
        self.increase_prune_plateau_count = increase_prune_plateau_count
        self.increase_prune_plateau_count_plateau_count = increase_prune_plateau_count_plateau_count
        self.evaluation_metric = evaluation_metric
        self.multi_class = multi_class

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the classifier on training data.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Training features. If not a DataFrame, it will be converted and feature
            names inferred/assigned.
        y : pandas.DataFrame or array-like, shape (n_samples, n_targets)
            Training targets. For binary single‑output tasks, a single column/array is
            expected. Multi‑label is supported via multiple columns.

        Returns
        -------
        None

        Notes
        -----
        - Internally constructs a :class:`BanditNRNClassifier` using the MIC‑based
          policy initialization and trains it with AdamW and cosine‑annealing restarts.
        - Early stopping and periodic pruning are controlled by the corresponding
          hyperparameters.
        """
        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()

        if not isinstance(y, pd.DataFrame):
            y = self._handle_non_dataframe_targets(y)
        y = y.copy()

        X = self._handle_empty_feature_names(X)

        # create textual class names if they are not given by user
        if self.target_names is None:
            self.target_names = [f"Class {i}" for i in range(y.shape[1])]
            y.columns = self.target_names

        # ecode data in necessary format
        X = self._fit_transform_encode_data(X)
        if self.binarization:
            X = self._fit_transform_binarize_features(X, y)
        feature_names = X.columns

        # pytorch data
        dataset = SimpleDataset(X.values, y.values)
        train_dl, train_holdout_dl = self._generate_training_data_loaders(dataset)

        # initial bandit policy
        assert len(feature_names) == X.shape[1], f"feat names: {len(feature_names)}; x: {X.shape[1]}"
        mic_c_policy, _ = compute_mic_matrix(X, y, alpha=.45, c=6)
        mic_c_policy = torch.tensor(mic_c_policy)

        if X.shape[1] < self.n_selected_features_input:
            Warning(
                "The number of features is less than 'n_selected_features_input'.  Using number of features instead.")
            self.n_selected_features_input = X.shape[1]

        # init model
        self.model = BanditNRNClassifier(
            target_names=self.target_names,
            feature_names=list(feature_names),
            input_size=len(feature_names),
            output_size=len(self.target_names),
            layer_sizes=self.layer_sizes,
            n_selected_features_input=self.n_selected_features_input,
            n_selected_features_internal=self.n_selected_features_internal,
            n_selected_features_output=self.n_selected_features_output,
            perform_prune_quantile=self.perform_prune_quantile,
            ucb_scale=self.ucb_scale,
            prune_strategy=self.prune_strategy,
            normal_form=self.normal_form,
            delta=self.delta,
            bootstrap=self.bootstrap,
            swa=self.swa,
            add_negations=self.add_negations,
            weight_init=self.weight_init,
            logits=self.logits,
            policy_init=mic_c_policy
        )

        optimizer = optim.AdamW(self.model.rn.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.t_0, T_mult=self.t_mult)
        trainer = BanditNRNTrainer(
            model=self.model,
            loss_func=self.loss_func(),
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=self.epochs,
            accumulation_steps=1,
            l1_lambda=0.,
            early_stopping_plateau_count=self.early_stopping_plateau_count,
            perform_prune_plateau_count=self.perform_prune_plateau_count,
            increase_prune_plateau_count=self.increase_prune_plateau_count,
            increase_prune_plateau_count_plateau_count=self.increase_prune_plateau_count_plateau_count,
            lookahead_steps=self.lookahead_steps,
            lookahead_steps_size=self.lookahead_steps_size,
            augment=None,
            augment_alpha=0.
        )

        # train model
        # The trainer defaults to optimizing the validation roc_auc_score.
        # To optimize against a different metric pass the sklearn metric to the 'evaluation_metric' parameter
        trainer.train(train_dl, train_holdout_dl, evaluation_metric=self.evaluation_metric,
                      multi_class=self.multi_class)
        trainer.set_best_state()

    # def predict(self, X: pd.DataFrame) -> pd.DataFrame:
    def predict(self, X: pd.DataFrame, decision_boundary=0.5) -> pd.DataFrame:
        """
        Predict class labels for the provided samples.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Untransformed input features. If not a DataFrame, it will be converted.
        decision_boundary : float, default=0.5
            Threshold applied to predicted probabilities to obtain class labels.
        
        Returns
        -------
        pandas.DataFrame
            Binary predictions with one column per class (or target). Column names
            match ``target_names`` when available.

        Raises
        ------
        AssertionError
            If the model has not been fitted.
        """
        assert self.model is not None, "must fit before prediction"

        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()

        X = self._handle_empty_feature_names(X)

        X = self._encode_prediction_data(X)
        dataset = SimpleDataset(X.values, np.ones(shape=(X.shape[0], self.model.output_size)))
        prediction_dl = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, num_workers=self.num_workers
            # very important to optimize these settings in production
        )

        predictions, _ = self.model.predict(prediction_dl)
        # class_predictions = (predictions > predictions.median()).astype(int)
        class_predictions = (predictions > decision_boundary).astype(int)

        class_predictions.rename(columns=lambda x: x.replace("probs_", ""), inplace=True)

        return class_predictions

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities for the provided samples.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Untransformed input features. If not a DataFrame, it will be converted.

        Returns
        -------
        pandas.DataFrame
            Probabilities of shape ``(n_samples, n_classes)`` with one column per
            class (or target). Column names match ``target_names`` when available.

        Raises
        ------
        AssertionError
            If the model has not been fitted.
        """
        assert self.model is not None, "must fit before prediction"

        if not isinstance(X, pd.DataFrame):
             X = self._handle_non_dataframe_features(X)
        X = X.copy()

        X = self._handle_empty_feature_names(X)

        X = self._encode_prediction_data(X)
        dataset = SimpleDataset(X.values, np.ones(shape=(X.shape[0], self.model.output_size)))
        prediction_dl = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, num_workers=self.num_workers
            # very important to optimize these settings in production
        )

        predictions, _ = self.model.predict_proba(prediction_dl)

        predictions.rename(columns=lambda x: x.replace("probs_", ""), inplace=True)

        return predictions

    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Compute the evaluation metric on given test data and labels.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Untransformed input features. If not a DataFrame, it will be converted.
        y : pandas.DataFrame or array-like, shape (n_samples, n_targets)
            Ground‑truth targets aligned with ``X``.

        Returns
        -------
        float
            The score produced by ``evaluation_metric`` (ROC‑AUC by default).

        Raises
        ------
        AssertionError
            If the model has not been fitted.
        """
        assert self.model is not None, "must fit before scoring"

        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()

        if not isinstance(y, pd.DataFrame):
            y = self._handle_non_dataframe_targets(y)
        y = y.copy()

        X = self._handle_empty_feature_names(X)

        X = self._encode_prediction_data(X)

        dataset = SimpleDataset(X.values, y.values)

        prediction_dl = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, num_workers=self.num_workers
            # very important to optimize these settings in production
        )

        predictions, _ = self.model.predict_proba(prediction_dl)

        return self.evaluation_metric(y.values, predictions.values)

    def _get_instance_params(self):
        """
        Return a dictionary of instance hyperparameters.

        Returns
        -------
        dict
            A mapping from hyperparameter names to their current values.
        """
        return {
            'layer_sizes': self.layer_sizes,
            'n_selected_features_input': self.n_selected_features_input,
            'n_selected_features_internal': self.n_selected_features_internal,
            'n_selected_features_output': self.n_selected_features_output,
            'perform_prune_quantile': self.perform_prune_quantile,
            'ucb_scale': self.ucb_scale,
            'prune_strategy': self.prune_strategy,
            'delta': self.delta,
            'bootstrap': self.bootstrap,
            'swa': self.swa,
            'add_negations': self.add_negations,
            'weight_init': self.weight_init,
            'logits': self.logits,
            'loss_func': self.loss_func,
            'perform_prune_plateau_count': self.perform_prune_plateau_count,
            'increase_prune_plateau_count': self.increase_prune_plateau_count,
            'increase_prune_plateau_count_plateau_count': self.increase_prune_plateau_count_plateau_count,
            'evaluation_metric': self.evaluation_metric,
            'multi_class': self.multi_class,
        }

    def explain_sample(
            self,
            X: pd.DataFrame,
            sample_index: int = 0,
            quantile: float = 1.0,
            decision_boundary: float = 0.5
    ) -> str:
        """
        Generate a local, human‑readable explanation for a single sample.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Input features; if not a DataFrame, it will be converted. Columns (or
            provided ``feature_names``) should match those seen during training.
        sample_index : int, default=0
            Row index within ``X`` to explain.
        quantile : float, default=1.0
            Fraction (0, 1] of the model’s most important logic to include.
        decision_boundary : float, default=0.5
            Threshold used to convert probabilities to class decisions in the text.

        Returns
        -------
        str
            A textual explanation describing which (possibly binarized) features and
            logical clauses influenced the prediction.

        Raises
        ------
        AssertionError
            If ``multi_class`` is True (unsupported) or if the model has not been fitted.

        Notes
        -----
        Explanations include value bounds when binarization is disabled; when enabled,
        explanations refer to thresholded (tree‑derived) feature predicates.
        """
        assert not self.multi_class, "explanation not currently supported for multi-class"
        assert self.model is not None, "must fit before explaining"

        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()

        X = self._handle_empty_feature_names(X)

        X = self._encode_prediction_data(X)

        self.min_max_features_dict = {
            col: {'min': self.numeric_features.iloc[:, i].min(), 'max': self.numeric_features.iloc[:, i].max()}
            for i, col in enumerate(self.numeric_features.columns)
        }

        dataset = SimpleDataset(X.values, np.ones(shape=(X.shape[0], 1)))

        return self.model.explain_samples(
            dataset[sample_index]['features'].unsqueeze(0),
            quantile=quantile,
            target_names=self.target_names,
            explain_type='both',
            sample_explanation_prefix="The prediction is in the",
            print_type='logical',
            ignore_uninformative=True,
            rounding_precision=3,
            show_bounds=not self.binarization,
            decision_boundary=decision_boundary,
            simplify=True,
            exclusions=None,
            min_max_feature_dict=self.min_max_features_dict,
            feature_importances=False
        )
