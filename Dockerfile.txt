From continuumio/anaconda3:master
WORKDIR /home/app
Copy . /home/app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["uvicorn", "--reload", "app:app"]