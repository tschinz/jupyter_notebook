# Basic Data Analysis
[![pipeline status](https://gitlab.com/smart-process-lab/basicdataanalysis/badges/master/pipeline.svg)](https://gitlab.com/smart-process-lab/basicdataanalysis/badges/master/pipeline.svg)
[![code coverage status](https://gitlab.com/smart-process-lab/basicdataanalysis/badges/master/coverage.svg)](https://gitlab.com/smart-process-lab/basicdataanalysis/badges/master/coverage.svg)
## Abstract
This Repo is a collection of basic initial data analysis: a form of data "unboxing". We typically get given initial data exports from customers followed by a simple verbal "have a look". This repo is a collection of those initial analyses.
Later, we typically request data in specific format with much larger data set, or even connect directly to their data source and extract as we see fit. In such cases the data handling is typically moved to a dedicated repo
The data analyses presented in this repo normally take the form of a Jupyter notebook: This allows a good mix of textual explanation and python code as well as output graph on a single page

## Getting Started
Clone this repository locally from the appropriate git source server. Explore each subdirectory as required. Open and run Jupyter notebook to rerun the analyses. Contribute with your own data set analyses.

### Prerequisites
An Anaconda installation will provide all the required tools, see [www.anaconda.com/distribution/](https://www.anaconda.com/distribution/) for a complete guide

### Installing
To install, clone the repository locally
```
git clone ...
```

## Running the tests
There is a simple automated (using GitLab CI) code quality control on all iPython Notebooks in this repo.

### And coding style tests
Automated coding style verifications is enforced using flake8

## Deployment
Clone the Repository locally and explore each subfolder as needed. To implement your own data analysis create a relevant subfolder for your implementation.

## Built With
* [Python](https://www.python.org/) - The Python programming language
* [Anaconda](https://https://www.anaconda.com/) - The Python and R distribution and package management for scientific computing
* [Jupyter](https://jupyter.org/) - The open document format that contain code, narrative text, equations and rich output
* [Gitlab CI](https://about.gitlab.com/product/continuous-integration/) - The Continuous Integration tool suite available in GitLab

## Contributing
Please read [CONTRIBUTING.md]() for details on our code of conduct, and the process for submitting merge requests to us.

## Versioning
We use [Semantic Versioning](http://semver.org/). For the list of versions available for this repo, see the [tags on this repository](). 

## Authors
* **Jerome Corre** - *Initial work* - [Gitlab Profile](https://gitlab.com/JeromeC47)

See also the list of [contributors](https://gitlab.com/groups/smart-process-lab/-/group_members) who participated in this project.

## Known Issues / TODOs
None for the moment!

## License
Copyright (C) 2019 HES-SO Valais-Wallis - All Rights Reserved
Unauthorized copying of this repository, or any of it file, via any medium is strictly prohibited
Proprietary and confidential

## Acknowledgments
* **Jerome Corre** - *Initial work, etc* - [Gitlab Profile](https://gitlab.com/JeromeC47) 
* **Michael Clausen** - *Gitlab group setup, reviews, etc* - [Gitlab Profile](https://gitlab.com/cm0x4d)
* **Silvan Zahno** - *Reviews, merge requests, etc* - [Gitlab Profile](https://gitlab.com/tschinz)
