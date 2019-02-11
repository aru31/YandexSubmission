<h1> Yandex Hackathon </h1>

This competition tackled the real life problem of identification of sub-atomic particles.

<h3> Our Procedure </h3>

1. Firstly, we divided our whole dataset into various different files on account of the huge size of the dataset.
2. After that, we started out with the preprocessing and EDA part.
3. For the EDA, we plotted various curves seeing correlation between the various variables among themselves and also with the variable that was to be predicted in the test dataset.
4. Through carefully analyzing, we removes various columns which were of very little importance to the model, and also used PCA so as to reduce the redundancy of various variables by summarizing the useful info from various correlated features.
5. Also after that StandardScaling was applied.
6. After this we trained our model using the tree based models. After various hit and trials, we stumbled upon some of the better algorithms for learning this particular task
