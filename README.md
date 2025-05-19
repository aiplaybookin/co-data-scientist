# co-data-scientist

Agentic data scientist who helps perform data analysis and train machine learning models.

# Pre-requisite

- Python 3.10.16
- Anthropic API keys

# Steps to run :

Clone the repo

```
git clone https://github.com/aiplaybookin/co-data-scientist.git
```

Install all the dependencies

```
pip install -r requirements.txt
```

Rename the .env.sample file to .env and update the file with your anthropic api keys

Run the app

```
chainlit run app_ds.py -h
```

# Sample questions

```
Use the file at <PATH TO YOUR FILE>/data/h1n1_dataset.csv and create a classification model using h1n1_vaccine as target label.
```
