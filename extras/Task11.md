# Task 11: Model Drift Strategy

Define two distinct types of model drift: Covariate Drift (input data distribution changes) and Concept Drift (relationship between inputs and target changes). For each type, describe the specific monitoring metrics and thresholds you would set up in the production environment to detect it.

## Analysis 

The Task 11 describes the working of a "Health Monitor" of a ML production system. It's commonly known that in ML, a model is at its peak performance the moment the training is finished; after that, it begins to "rot" as the real world changes.

To keep a 99% Recall intact, it's necessary to distinguish between the data changing and the world changing.

### 1. Covariate Drift ("Input" Problem)

This happens when the distribution of the input features (X) changes, but the underlying relationship with the target remains the same. Essentially, a model is seeing "new" types of data it wasn't trained on.

As an example, let's suppose that a new brand of vibration sensor is installed, and it is more sensitive. The raw values jump from an average of 2.0 Hz to 4.5 Hz. The machine isn't failing yet, but the data "looks" different to the model.

**Monitoring Metric:** Population Stability Index (PSI) or Jensen-Shannon Divergence (JSD).

Possible thresholds:
- $PSI < 0.1$: No significant change.
- $0.1 ≤ PSI < 0.25$: Slight drift detected (Alert maintenance to check sensor calibration).
- $ PSI ≥ 0.25 $: Significant drift (Trigger automatic feature scaling or retraining).

$$PSI = \sum_{i=1}^{B} \left( (Actual\_i - Expected\_i) \times \ln\left(\frac{Actual\_i}{Expected\_i}\right) \right)$$

### 2. Concept Drift ("Logic" Problem)

If the inputs ($X$) might look the same, but the relationship between those inputs and the target ($Y$) has changed. The model's "concept" of what a failure looks like is now wrong.

As an example, let's say that due to extreme summer heat in the laundry facility, a vibration of 3.0 Hz (which used to be "Healthy") now leads to a pump failure because the metal is more brittle. The inputs haven't changed, but the meaning of the inputs has.

**Monitoring Metric:** Recall Degradation (since our priority is preventing downtime) and F1-Score.

Thresholds:
- $Recall > 95%$: System Healthy.
- $Recall < 90%$: Warning — The model is missing failures it used to catch.
- $Recall < 85%$: CRITICAL DRIFT -  Immediate trigger for "Challenger" model promotion and emergency retraining.
