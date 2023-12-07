from django.db import models


class CsvFile(models.Model):
    file = models.FileField(upload_to='csv/')
