from flask import Flask
from flask import render_template, request
import pandas as pd
import json
from pca import pca_function
from nn import NN
from gen_set import gen_set_function
from svmpca import svmpca_function
from tsne import tsne_function

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/donorschoose/projects")
def donorschoose_projects():
    return json.dumps({'a':1})

@app.route('/background_process_test')
def background_process_test():
    print("Hello")
    return json.dumps({'b':1})

@app.route('/my')
def return_csv():
    return pd.read_csv('2d.csv').to_json()
    #return json.dumps({'b':1})

@app.route('/', methods=['POST'])
def read_data():
    filename = request.form['filename']
    epochs = request.form['epochs']
    embedding_type = request.form['embed']
    algo = request.form['algo']
    b_size = request.form['b_size']
    #margin = request.form['margin']
    relation = 'sports' #citytown
    weights = request.form['weights']
    w = []
    for i in weights.split():
        w.append(float(i))

    if algo=='nnpro':
        gen_set_function(filename,relation)
        n = NN()
    #data = pca_function(text,'language')
    #data = {'csv_data': data}
        data = n.nn_function('gen_set.pkl',epochs,0,embedding_type,int(b_size),w,1)
    elif algo=='pca':
        data = pca_function(filename,relation)
    elif algo=='svmpca':
        data = svmpca_function(filename,relation)
    elif algo=='tsne':
        data = tsne_function(filename,relation)
    data.to_csv('static/2d.csv',index=False)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)