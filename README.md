# News-Text-Classifier

## Overview
* Simple text classifier for news blurbs
* Trained using TensorFlow 2.0
* [model.py](https://github.com/mikepatel/News-Text-Classifier/blob/master/model.py) for model definitions
* [parameters.py](https://github.com/mikepatel/News-Text-Classifier/blob/master/parameters.py) for model hyperparameters
* [train.py](https://github.com/mikepatel/News-Text-Classifier/blob/master/train.py) for preprocessing and model training

## Data
* Dataset available [here](https://github.com/mikepatel/News-Text-Classifier/blob/master/data/bbc-text.csv)

## Instructions
```
python train.py
```

## Results
* Test loss: 0.136383
* Test accuracy: 0.955157
* Precision: 0.958716
* Recall: 0.937220

### Predictions
| News text | Prediction |
:-----------|:------------
| The US and the Taliban have signed a "comprehensive peace agreement" to end fighting in Afghanistan after more than 18 years of conflict. | politics |
| Officials on the US West Coast have reported three unexplained coronavirus cases, raising concerns the virus could be spreading within the community. | business |
| The Democratic race moves south later on Saturday as voters in South Carolina have their say on who should be the party's White House nominee. | sport |
| Several actresses have walked out of the César awards ceremony in Paris after Roman Polanski, who was convicted of the statutory rape of a 13 year old in 1977, won best director. | entertainment |
| After all the slick presentations, after David Beckham has swept in and charmed the 40th floor of a Manhattan hotel, it is hard to believe that, not so long ago, Major League Soccer came close to falling apart. | sport |
| March sees the release of Onward and the Mulan reboot, as well as Hirokazu Kore-Eda’s French-language debut, The Truth. | entertainment |
| In 2015, the boss of a card payments company in Seattle introduced a $70,000 minimum salary for all of his 120 staff - and personally took a pay cut of $1m. | business |

### Visualization
![25 February 2020](https://github.com/mikepatel/News-Text-Classifier/blob/master/results/25-02-2020_18-28-21/Training%20Accuracy.png)
