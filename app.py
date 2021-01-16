from flask import Flask, request
app = Flask(__name__)
import pandas as pd
import pickle
import Process as pr
@app.route('/')
def myfunct():
    return 'Hello'
@app.route('/getPredict',methods=['POST','GET'])
def hello_world():
    Cloth=request.args.get('Cloth')
    Food=request.args.get('Food')
    Fuel=request.args.get('Fuel')
    Holiday=request.args.get('Holiday')
    Home=request.args.get('Home')
    Kids=request.args.get('Kids')
    Pharm=request.args.get('Pharm')
    Shopping=request.args.get('Shopping')
    Transport=request.args.get('Transport')
    pred = pr.getPrediction(Cloth,Food,Fuel,Holiday,Home,Kids,Pharm,Shopping,Transport)
    return str(pred)