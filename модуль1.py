from flask import Flask, render_template, request
  
import os 


app = Flask(__name__)
   
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
   
@app.route('/')
def show_main_page():
    return render_template('file.html')
   
@app.route('/upload', methods=['POST'])
def handle_file_upload():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("file")
        #amount = request.files.getlist("file").count()
        for uploaded_file in uploaded_files:
            filename = uploaded_file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
        file = open('uploads/test.txt', 'r')
        content = file.read()
        
        file.close()
        return content
   
if __name__ == '__main__':
    app.run(debug=True)