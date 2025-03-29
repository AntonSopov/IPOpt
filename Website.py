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

from operator import itemgetter
from operator import attrgetter
  
import os 




app = Flask(__name__)
dash = Dash(__name__, server=app, url_base_pathname='/results/')#Эта штука для создания даша внутри фласка по пути /results/
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
font = {'font-size' : '2rem'}
   


dash.layout = [#Эта штука для создания тела дашы
    html.Div(children='Результаты работы алгоритма'),
    html.Hr(),
    html.Div(id='none',children=[],style={'display': 'none'}),
    dcc.Graph(figure={}, id='graph'),
    html.Pre(id='click-data')
]


@dash.callback(#Штука что обновляет график для какой то функции, в данном случае func
    Output(component_id='graph', component_property='figure'),
    Input(component_id='none', component_property='value')
)
def update_graph(col_chosen):
    for i in range(len(Gl.solutions)):
        Gl.sorting_indices.append([i, Gl.solutions[i]['Прибыль']]) 
    Gl.sorting_indices.sort(key = lambda x: x[1])
    x = []
    y = []
    for i in range(len(Gl.solutions)):
        x.append(Gl.solutions[Gl.sorting_indices[i][0]]['Прибыль'])
        y.append(Gl.solutions[Gl.sorting_indices[i][0]]['Риск'])
    
    fig = px.line(x=x, y =y, title="Наведите на точку на графике, чтобы посмотреть информацию о решении.", markers=True, labels={'x': 'Прибыль', 'y':'Риск'})
    #fig.update_traces(line=dict(color="green", width=2.5)) # настройка стиля кривой
    fig.update_traces(marker=dict(size=10))
    return fig


@dash.callback(#Эта штука выводит данные о клике на графике
    Output('click-data', 'children'),
    Input('graph', 'clickData'))
def display_click_data(clickData):
    click_index = Gl.sorting_indices[clickData['points'][0]['pointIndex']][0]
    print(click_index, "  ", clickData['points'][0]['pointIndex'])
    solution_text_info = "\tИнформация о решении:\n\n\tПрибыль: " + str(round(Gl.solutions[click_index]["Прибыль"], 4)) + "\n\tРиск: " + str(round(Gl.solutions[click_index]["Риск"], 4))  
    solution_text_info += "\n\n\tСписок проектов, входящих в портфель:\n"
    prj_ctr = 0
    for i in range(len(Gl.problem._project_names)):
        solutions_present = 0
        solution_text_info += "\n\tГруппа проектов №" + str (i+1) + ':\t'
        for j in range(len(Gl.problem._project_names[i])):
            if Gl.solutions[click_index]["Портфель"][prj_ctr] == 1:
                solution_text_info += str(Gl.problem._project_names[i][j]) + '  '
                solutions_present = 1
            prj_ctr +=1
        if solutions_present == 0:
            solution_text_info += "-"
    return solution_text_info
    #return json.dumps(clickData, indent=2)



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

        Gl.solutions = copy.deepcopy(Gl.problem.optimize('nsga2', opt_params))

        return redirect("/results")


if __name__ == '__main__':
    app.run(debug=True)