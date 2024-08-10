from flask import Flask, request, render_template
import pandas as pd

from source.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            city=request.form.get('city'),
            state=request.form.get('state'),
            zip=request.form.get('zip'),
            lat=float(request.form.get('lat')),
            long=float(request.form.get('long')),
            city_pop=int(request.form.get('city_pop')),
            job=request.form.get('job'),
            unix_time=int(request.form.get('unix_time')),
            category=request.form.get('category'),
            amt=float(request.form.get('amt')),
            merchant=request.form.get('merchant'),
            merch_lat=float(request.form.get('merch_lat')),
            merch_long=float(request.form.get('merch_long')),
            trans_year=int(request.form.get('trans_year')),
            trans_month=int(request.form.get('trans_month')),
            trans_day=int(request.form.get('trans_day')),
            trans_hour=int(request.form.get('trans_hour')),
            trans_minute=int(request.form.get('trans_minute')),
            trans_second=int(request.form.get('trans_second')),
            day_of_week=int(request.form.get('day_of_week')),
            distance_to_merchant=float(request.form.get('distance_to_merchant')),
            age=int(request.form.get('age'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")   