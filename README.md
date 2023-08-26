# [Visualized Monitoring of Generative AI Model Drift](https://youtu.be/t9cYCfqrPJw?si=C7jcCI5IslC5vgx5) x Conformal Prediction
So this is a companion to introduce the concept from the following [output monitoring project](https://github.com/rabbidave/Squidward-Tentacles-and-Spying-on-Outputs-via-Conformal-Prediction) which does the following:

Summary: [Squidward](https://github.com/rabbidave/Squidward-Tentacles-and-Spying-on-Outputs-via-Conformal-Prediction) takes incoming messages and does stepwise comparison of their log-likelihood to a given baseline, such that we can compute a pseudo-confidence interval, and use that for appending (or not appending) our language model output

It's predicated on the concept of [conformal prediction](https://github.com/valeman/awesome-conformal-prediction)

## Note:

Different metrics of non-conformity are more appropriate depending on the situation; for the moment I intend to use [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) as derived from [log-likelihood](https://www.statisticshowto.com/log-likelihood-function/) thereby allowing the system to effectively monitor data drift and/or eventually detect the emergence of new clusters in a vector space (forthcoming project)

In lieu of a requirements.txt I've included a list of packages below that I remember needed when I built the virtual environment.

Also the code outputs pngs and I've also uploaded a [video to youtube](https://youtu.be/sAwZhlePgAc)

### Dependencies

pip install numpy matplotlib umap-learn        