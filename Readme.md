# Multilevel Stacked Generalization

Stacked generalization (or stacking) (Wolpert, 1992) is a different way of combining multiple models, that introduces the concept of a meta learner. Although an attractive idea, it is less widely used than bagging and boosting. Unlike bagging and boosting, stacking may be (and normally is) used to combine models of different types. The procedure is as follows:

1. Split the training set into disjoint N sets.
2. Train base learner on N-1 sets and predict on the left out set.
3. Repeat until all sets were left out once.
4. Either take average of all predictions or simply feed all predictions from 3) as input for another layer of base learners.
5. Repeat procedure for arbitrary levels.






