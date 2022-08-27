import pandas as pd
import numpy as np
import pickle

from flask import Flask,render_template,request,jsonify

with open('ins_model.pkl','rb') as file:
    rf=pickle.load(file)

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ins_detail', methods=['POST'])
def ins_detail():
    user_name = request.form['u_nm']
    user_age = request.form['u_age']
    user_bmi= request.form['u_bmi']
    user_child= request.form['u_child']
    user_smoke= request.form['u_smoker']
    user_Gender = request.form['u_gd']
    user_mob = request.form['u_mb']
   
    print(f"Data By User Input ={user_name}")
    print(f"Data By User Input ={user_age}")
    print(f"Data By User Input ={user_bmi}")
    print(f"Data By User Input ={user_child}")
    print(f"Data By User Input ={user_smoke}")
    print(f"Data by user input={user_Gender}")
    print(f"Data by user input={user_mob} ")

    user_input=np.array([user_age,user_bmi,user_child,user_smoke],ndmin=2)
    result=rf.predict(user_input)
    print(result)
    return  render_template('result.html',res=result[0],name=user_name,mb=user_mob,gn=user_Gender,ag=user_age,bm=user_bmi,ch=user_child)

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080,debug=False)

