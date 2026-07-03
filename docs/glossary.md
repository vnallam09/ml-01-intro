# Glossary

Use this page to record terms and ideas that help you understand
professional analytics projects.

This project covers characterizing machine learning problems:
identifying the task type, selecting a target, and understanding
what kind of model would be appropriate.

Pro-tip: Expand the VS Code **Outline** view (below the navigator on the right)
to see this file organization at-a-glance.

## Project Organization

### source code

Source code are instructions that tell the computer what to do.
In a Python project, source code lives in files ending with `.py`.

### module

A module is one Python file that contains related code.
A module may include constants, functions, imports, and a `main()` function.
A project may have many modules working together.

### package

A package is a folder of related Python modules.
A package usually includes an `__init__.py` file.
The init file allows code in that folder to be
imported and reused across a project.

### notebook

A notebook is an interactive file used to combine
code, output, notes, and narrative.
Notebooks are useful for exploration, experiments,
and explaining analysis step by step.

## Reuse and Workflow

### reusable function

A reusable function is a named block of code that performs
one clear task and can be called more than once.
Good functions make projects easier to read, test, debug, and modify.

### dependency

A dependency is an external package or tool that a project
needs in order to run.
Dependencies are listed in `pyproject.toml`
and the environment can be easily recreated using `uv`.

### workflow

A workflow is an ordered process for completing work.
In a project, a workflow might include running code,
checking results, making changes, testing again,
and saving progress with Git.

## Machine Learning Concepts

### machine learning

Machine learning is a way of building programs that learn patterns
from data rather than following hand-written rules.
The program improves its predictions by seeing examples.

### supervised learning

Supervised learning is a type of machine learning where the training data
includes a target column - a known answer the model learns to predict.
You choose the target; that choice defines the problem.

### unsupervised learning

Unsupervised learning is a type of machine learning where there is no target column.
The model explores structure in the data without being told what to look for.
Clustering is a common unsupervised approach.

### target variable

The target variable is the column a supervised model is trained to predict.
Choosing the target is an analyst decision that shapes the entire modeling approach.

### features

Features are the input columns used to make a prediction.
The analyst decides which features to include, transform, or remove.

### classification

Classification is a supervised ML task where the target is a category.
The model learns to assign inputs to one of a fixed set of classes.
Predicting whether an email is spam or not spam is a classification problem.

### regression

Regression is a supervised ML task where the target is a continuous number.
The model learns to predict a quantity rather than a category.
Predicting a house price from its features is a regression problem.

### clustering

Clustering is an unsupervised ML task where the model groups similar
observations together without being given labels.
The analyst interprets what each group means.

### discrete variable

A discrete variable takes one of a limited set of distinct values.
Category columns such as species, color, or rating are discrete.
A discrete target usually means a classification problem.

### continuous variable

A continuous variable can take any numeric value within a range.
Columns such as temperature, price, or weight are continuous.
A continuous target usually means a regression problem.
