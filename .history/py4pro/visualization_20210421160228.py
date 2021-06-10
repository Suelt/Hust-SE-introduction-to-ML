import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

credit = pd.read_csv("C:\\pyproject\\py4pro\\transformed.csv")

col_dicts = {}
cols = ['status','credit_history', 'purpose', 'savings', 'employment_duration','installment_rate', 'personal_status_sex', 'other_debtors',
        'present_residence','property','other_installment_plans','housing','number_credits','job','people_liable','telephone','foreign_worker']
