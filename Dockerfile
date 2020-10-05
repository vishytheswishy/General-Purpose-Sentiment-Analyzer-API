  FROM python:3.6-slim-stretch
  LABEL maintainer="Vishaal Yalamanchali, vyalaman@uci.edu"
  RUN apt update
  RUN apt install -y python3-dev gcc

  ADD nltkmodels.py nltkmodels.py
  ADD vectoriser.pkl vectoriser.pkl
  ADD LR.pkl LR.pkl
  ADD template template
  ADD static static
  ADD template/input.html template/input.html
  ADD static/mystyle.css static/mystyle.css
  ADD requirements.txt requirements.txt
  ADD app.py app.py

  RUN pip3 install -r requirements.txt
  RUN python3 nltkmodels.py
  RUN ls

  EXPOSE 5000
  CMD [ "python3", "app.py" ,"runserver", "-h", "0.0.0.0"]]