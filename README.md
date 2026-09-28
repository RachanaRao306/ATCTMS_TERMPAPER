# ATCTMS_TERMPAPER



The datasets used are Global_opinion qa and Opinions QA. The data for global opinion is already in data folder. But could not upload the opinions qa because of size. I have attached the link for opinions qa. The only thing to change in opinions qa is the naming for human response. Go to human_resp and in each wave there is a response.csv, copy all to temp folder and rename them as response_W{number}.csv

Run programs in the following order.

1. global_opinions.py
2. opinionsqa.py
3. Opinion_global_QM.ipynb (in colab)
4. globalopinion_valuation.py
5. opinionsqa_valuation.py
6. model_comp.py
7. generate_graphs.py
8. 3model_graph.py

Note: only issue is the reading of csv files and naming convention. It is not curated and requires attention.
Python packages to run above programs
1. transformers
2. torch
3. pandas
4. matplotlib
5. scipy
6. huggingface_hub
7. accelerate
