from django import forms


class CsvUploadForm(forms.Form):
    file = forms.FileField(label='CSV File')