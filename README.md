# Emerging Technologys Module Tasks

## Introduction

Four tasks were undertaken in the assessment for Emerging Technologies 2020 in a Juypter Notebook.

## Viewing the project

Click the binder link to view the Juypter Notebook: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/shaunamoffatt/EmergingTechnologyTasks/main)

#### Installing Jupyter Notebook

You need to download Anacondas latest [Python 3] (https://www.anaconda.com/products/individual) (version used conda 4.9.2)

You can follow instructions [here](https://jupyter.readthedocs.io/en/latest/install/notebook-classic.html) to install juyper notebook

or 

```
pip install notebook
```

Full list of [requirements](requirements.txt)


## [Square Root of 2 in Python](Task_1.ipynb)
### Task 1

The square root of 2 was calculated in Python using Newtons Rhapson Method for calculating the square. Results were compared to pythons funtion for the squareroot 'math.sqrt(2)' in 'math' library.

Packages Included :

```
import matplotlib.pyplot as plt
import math
import decimal
```

## [Chi-squared test for independence](Task_2.ipynb)
### Task 2

The library 'scipy.stats' was used to perform the Chi-squared test for Independence on a dataset. The dataset consisted of a sample of 650 from a city with a population of 1,000,000 that was split into 4 neighbourhoods: A, B, C and D where everybodies occupation was categorized into 1 of 3 brackets: "White collar", "Blue collar" and "No collar".(Data retrieved from [wikipedia] (https://en.wikipedia.org/wiki/Chi-squared_test)

Packages Included :

```
from scipy.stats import chi2_contingency as chi
from IPython.core.display import HTML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import isclose
```

## [STDEV.P and STDEV.S in MS. Excel](Task_3.ipynb)
### Task 3

Microsoft Excels functions for standard deviation STDEV.P and STDEV.S where explained and compared when calculating the standard deviation of an arrays of numbers.

Packages Included :
```
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import statistics
import math
import random
from scipy.stats import norm
```

## [KNN for the Iris DataSet using scikit-learn](Task_4.ipynb)
### Task 4 

Using 'scikit-learn' the algorithm K-Nearest Neighbour was used on the famouse iris dataset, to train and test the dataset and allow for predictions to be made on given inputs of features.

Packages Included: 
``
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns
%matplotlib inline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
``