import plotly.graph_objects as go
import matplotlib.pyplot as plt

def printTable(classifier,precision,recall,accuracy):
    table= go.Figure(data=[go.Table(header=dict(values=['Classifier','Precision','Recall','Accuracy'],align=['left'],font=dict(size=13)),
    cells=dict(values=[classifier,precision,recall,accuracy],align='left',font_size=12))])
    table.show()
