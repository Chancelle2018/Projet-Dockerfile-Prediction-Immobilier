FROM jupyter/scipy-notebook


RUN pip install joblib

COPY House_Data.csv ./House_Data.csv


COPY Lab_2_house_prices.py ./Lab_2_house_prices.py

RUN python3 Lab_2_house_prices.py
