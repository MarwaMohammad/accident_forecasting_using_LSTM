From continuumio/anaconda3:master
WORKDIR /home/app
Copy . /home/app
RUN pip install -r requirements.txt
EXPOSE 8000:8000
CMD [ "python", "app.py" ]