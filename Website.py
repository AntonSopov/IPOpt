from turtle import title
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
import numpy as np

from dash import Dash, dcc, html
from dash.dependencies import Input, Output

import pandas as pd

import plotly.express as px


import json

import copy
import Global as Gl
from Nsga2 import *
from Problem import *

  
import os 




app = Flask(__name__)
dash = Dash(__name__, server=app, url_base_pathname='/results/')#Эта штука для создания даша внутри фласка по пути /results/
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
   


dash.layout = [#Эта штука для создания тела дашы
    html.Div(children='Результаты работы алгоритма'),
    html.Hr(),
    html.Div(id='none',children=[],style={'display': 'none'}),
    dcc.Graph(figure={}, id='graph'),
    html.Pre(id='click-data')
]



def func(x):
  #y = ma.sin(x)+ma.e**
 return -7*np.sin(x) + 10*np.e**np.cos(x)+0.8*x



@dash.callback(#Штука что обновляет график для какой то функции, в данном случае func
    Output(component_id='graph', component_property='figure'),
    Input(component_id='none', component_property='value')
)
def update_graph(col_chosen):
    x = np.arange(0, 5, 0.1)
    y = func(x)
    fig = px.line(x=x, y=y, title="Какая то функция")
    return fig


@dash.callback(#Эта штука выводит данные о клике на графике
    Output('click-data', 'children'),
    Input('graph', 'clickData'))
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)



@app.route('/')
def show_main_page():
    return render_template('file.html')


@app.route('/', methods=['POST'])
def main_page_reset():
    return render_template('file.html')
   
@app.route('/Settings', methods=['POST'])
def show_settings_page():
    if request.method == 'POST':
        # Getting file paths
        filePathes = []
        uploaded_files = request.files.getlist("file")
        
        # if no files are selected
        if uploaded_files[0].filename == '':
            return render_template('file.html')

        # otherwise
        for uploaded_file in uploaded_files:
            filename = uploaded_file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            filePathes.append(file_path)
            uploaded_file.save(file_path)
        


        # Forming an IPOpt problem
        ipopt_problem = Problem(filePathes, ',')
        Gl.problem = copy.deepcopy(ipopt_problem)
        
        headings, data = Gl.problem.transform_data()

    return render_template('settings.html', headings = headings, data = data)


@app.route('/Start', methods=['POST'])
def handle_file_upload():
    if request.method == 'POST':
        amount_maxFEs = int(request.form.get('fes'))
        amount_invBudget = float(request.form.get('invBudget'))
        amount_maxInc = float(request.form.get('maxInc'))
        amount_maxRisk = float(request.form.get('maxRisk'))

        amount_pop_size = int(request.form.get('pop_size'))
        amount_epslevel = float(request.form.get('epslevel'))
        amount_prob_crossover = float(request.form.get('prob_crossover'))
        amount_prob_mutation = float(request.form.get('prob_mutation'))

        # эта часть в разработке
        Gl.problem.set_parameters(investmentBudget = amount_invBudget, maxIncome = amount_maxInc, maxRisk = amount_maxRisk)

        Gl.num_objectives = 2
        Gl.population_size = amount_pop_size

        opt_params = {
            'population_size': Gl.population_size,
            'crossover_prob': amount_prob_crossover, 
            'mutation_prob': amount_prob_mutation,
            'max_eval': amount_maxFEs,
            'constraints': True,
            'eps': amount_epslevel,
            'filename': 'result_test'
        }
        Gl.problem.optimize('nsga2', opt_params)
        
        return redirect("/results")


if __name__ == '__main__':
    app.run(debug=True)