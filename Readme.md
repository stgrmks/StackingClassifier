# Multilevel Stacked Generalization

Stacked generalization (or stacking) (Wolpert, 1992) is a different way of combining multiple models, that introduces the concept of a meta learner. Although an attractive idea, it is less widely used than bagging and boosting. Unlike bagging and boosting, stacking may be (and normally is) used to combine models of different types. The procedure is as follows:

1. Split the training set into disjoint sets.
2. Train several base learners on the some parts.
3. Test the base learners on the left out part.
4. Using the predictions from 3) as the inputs, and the correct responses as the outputs, train a higher level learner.

Note that steps 1) to 3) are the same as cross-validation, but instead of using a winner-takes-all approach, we combine the base learners, possibly nonlinearly.

## Getting Started

Simply copy the folder into your working directory and import the class. See example.

### Prerequisites

What things you need to install the software and how to install them

```
Pandas
Numpy
```

## Example

A kaggle competition dataset was used to show the implementation of a multilevel stacker.



