import joblib
from django.shortcuts import render
from .forms import CsvUploadForm
import pandas as pd
from .preprocessing import preworking
from os import listdir
from os.path import join


def predict_price(request): 
    if request.method == 'POST': 
        form = CsvUploadForm(request.POST, request.FILES) 
        if form.is_valid(): 
            csv_file = request.FILES['file']
            df = pd.read_csv(csv_file, index_col='id')
 
            data_privat = preworking(df)
            
            # Загрузка обученной модели из файла plk 
            model_list = list()
            folder_path = 'prediction_app/models'
            for f in listdir(folder_path):
                file_path = join(folder_path, f)
                model_list.append(joblib.load(file_path))

            pred = []
            i = 0
            for model in model_list:
                print(i)
                predictions = model.predict(data_privat)
                pred.append(predictions)
                i+=1
 
            predictions = sum(pred) / len(pred)
 
            predictions_df = pd.DataFrame(predictions, columns=['price'])

            # Сохранение предсказаний в виде CSV файла 
            predictions_df.to_csv('predictions.csv', index=False)

            return render(request, 'prediction_app/result.html', {'form': form, 'predictions_df': predictions_df}) 
    else: 
        form = CsvUploadForm() 
        return render(request, 'prediction_app/upload.html', {'form': form})