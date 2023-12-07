# Naive Bayes Classifier
This is a Naive Bayes classifier written in Python. It uses the maximum likelihood estimation method to estimate the class conditional probabilities of a dataset. The classifier is based on the assumption of conditional independence between the attributes given the class.

## Getting started
### Prerequisites
Python 3.x
NumPy
SciPy

### Dataset
The watermelon dataset 3.0 is used as an example in this implementation. It includes the following attributes:

Color (discrete): Green, black, or white.
Root (discrete): Curled, slightly curled, or straight.
Sound (discrete): Clear, dull, or ringing.
Texture (discrete): Clear or slightly blurry.
Navel (discrete): Slightly hollow, flat, or concave.
Touch (discrete): Hard or soft.
Density (continuous): A float value.
Sugar ratio (continuous): A float value.
Class (discrete): Good or bad.
## Assumptions
### Multinomial distribution
In the case of discrete attributes, the class conditional probabilities are assumed to follow a multinomial distribution. The probabilities of each attribute's different values under each category are calculated using the maximum likelihood estimation method.

### Normal distribution
In the case of continuous attributes, the class conditional probabilities are assumed to follow a normal distribution. The mean and standard deviation of each attribute under each category are estimated using the maximum likelihood estimation method.

### License
This project is licensed under the MIT License - see the LICENSE file for details.