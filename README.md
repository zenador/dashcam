### Dashcam

Uses Plotly Dash to create flexible dashboards from tables with results (counts of true/false positives/negatives) of machine learning classifier models, allowing you to see how metrics like precision, recall, F1 score and accuracy change over time for different subcategories.

See `example.csv` for the format the results have to be stored in. You can easily use this with different models and different subcategories as long as the results are all stored in that same format. 

You can run a local server for this dash app by changing directory to this one and then `python run.py`, but follow these steps first:

1. Setup dependencies
- Use a virtual environment and `pip install -r requirements.txt`

2. Setup your db connection
- Fill in your database credentials in `config.py` and modify `get_db` in `db.py` to connect to your database. Alternatively, export your results in the correct format to replace `example.csv`.
