
--8<-- "CONTRIBUTING.md"

## Install from source

To install spurt from source, you will need:

- Python 3.9 or newer + pip

The latest source code can be installed using

```
$ pip install git+https://github.com/isce-framework/spurt.git
```

Alternatively, clone the repository and install the local copy using

```
$ git clone https://github.com/isce-framework/spurt.git
$ cd spurt
$ pip install .
```


## Running the test suite

To run the test suite, with pytest and pytest-cov installed

```
$ cd spurt
$ python3 -m pytest
```
