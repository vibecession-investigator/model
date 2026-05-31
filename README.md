This work aimed to reproduce the claims of bluesky user @kulp.bsky.social, who claimed that the user Will Stancil @whstancil.bsky.social had ignored the well known fact that:

"There is no vibecession if you model sentiment using U3, CPI, 30yr mortgage and the differences between CPI and Mortgage rates and their 5yr and 10yr averages"

Rather than share their work, this user instead gave a set of instructions to recapitulate it over a series of posts which can be viewed here:
https://skythread.mackuba.eu/?author=kulp.bsky.social&post=3mmhxs2yqtk2e
This repo is a faithful attempt to follow those instructions.

The results of this attempt have not reproduced this user's claims. A linear model using these variables was not able to predict sentiment from Q3 2022 onwards, despite fitting the training data fairly well. This supports Stancil's hypothesis that sentiment diverged from macroeconomic indicators after Q3 2022.

The claim by me (@baldzera.bsky.social) that real wages would make the model worse was also incorrect. It was made under the assumption that the claims made about the model excluding real wages were true, which does not seem to be the case.

A visualisation of both models:

<img width="1934" height="1369" alt="model_predictions" src="https://github.com/user-attachments/assets/3c8f7711-0bc7-4ebd-832c-92c6d568094f" />

Here, we use monthly data and set the training cutoff to February 2020 to ensure that none of the COVID-19 pandemic is included in the training set (for some reason):

<img width="1934" height="1368" alt="model_predictions" src="https://github.com/user-attachments/assets/d3526c51-8278-4c9f-bfa8-9f1bbed2bf72" />
