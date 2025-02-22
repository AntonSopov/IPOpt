from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import numpy as np

import copy
import Global as Gl
from Nsga2 import *
from Problem import *

  
import os 



app = Flask(__name__)
   
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
   
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

        return "Yay"


if __name__ == '__main__':
    app.run(debug=True)