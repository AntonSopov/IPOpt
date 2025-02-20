from flask import Flask, render_template, request
  
import os 


app = Flask(__name__)
   
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
   
@app.route('/')
def show_main_page():
    return render_template('file.html')
   
@app.route('/Start', methods=['POST'])
def handle_file_upload():
    if request.method == 'POST':
        filePathes = [] #Здесь будут все пути к загруженным файлам
        amountIndividuals = int(request.form.get('individuals'))#Хранение количества индивидов в поле
        amountGenerations = int(request.form.get('generations'))#Хранение количества индивидов в генерациях
        uploaded_files = request.files.getlist("file")
        for uploaded_file in uploaded_files:
            filename = uploaded_file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            filePathes.append(file_path)
            uploaded_file.save(file_path)

        return "Yay"
   
if __name__ == '__main__':
    app.run(debug=True)