# Code File Description

## predict.py

Prediction on all datasets using in-context learning. You can adjust the ``seed`` for reproduction and ``max_context`` for max context size. Prediction result will be stored in ``predict-ctx.csv`` under folder of each dataset.

## finetune.py

Finetune the model for each dataset respectively and make prediction file ``predict.csv`` and store finetuned model in folder ``models`` .

# Hyperparameters

For reproduction, we use $42$ as random seed for all tasks.

# Notes

Since ``tabpfn.finetuning`` use **v2.5** as fixed pretrained model, we hack the library code for using the same model of in-context learning without doing anything other things for improving preformance.

What we do is modify the ``_create_estimator`` method in ``FinetunedTabPFNRegressor``/``FinetunedTabPFNClassifier``：

From
```python
def _create_estimator(self, config: dict[str, Any]) -> TabPFNRegressor:
        """Create the TabPFNRegressor with the given config."""
        return TabPFNRegressor.create_default_for_version(
            version=ModelVersion.V2_5,
            **config,
            fit_mode="batched",
            differentiable_input=False,
        )
```
to
```python
def _create_estimator(self, config: dict[str, Any]) -> TabPFNRegressor:
        """Create the TabPFNRegressor with the given config."""
        return TabPFNRegressor.create_default_for_version(
            version=ModelVersion.V2_6,
            **config,
            fit_mode="batched",
            differentiable_input=False,
        )
```