from flask import Flask,request, render_template
import numpy as np
import pandas as pd

from src.pipelines.predict_pipeline import CustomData, PredictPipeline
application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        data=CustomData(
            Item_Identifier=request.form.get('Item_Identifier'),
            Item_Weight=request.form.get('Item_Weight'),
            Item_Fat_Content=request.form.get('Item_Fat_Content'),
            Item_Visibility=request.form.get('Item_Visibility'),
            Item_Type=request.form.get('Item_Type'),
            Item_MRP=request.form.get('Item_MRP'),
            Outlet_Establishment_Year=request.form.get('Outlet_Establishment_Year'),
            Outlet_Size=request.form.get('Outlet_Size'),
            Outlet_Location_Type=request.form.get('Outlet_Location_Type'),
            Outlet_Type=request.form.get('Outlet_Type'),
            Outlet_Age=request.form.get('Outlet_Age')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)