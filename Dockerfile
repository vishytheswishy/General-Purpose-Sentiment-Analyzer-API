  FROM python:3.6-slim-stretch
  LABEL maintainer="Vishaal Yalamanchali, vyalaman@uci.edu"
  RUN apt update
  RUN apt install -y python3-dev gcc
  ADD nltkmodels.py nltkmodels.py
  RUN ls
  ADD vectoriser.pkl vectoriser.pkl
  ADD LR.pkl LR.pkl
  ADD tokenization.py tokenization.py
  ADD template/input.html template/input.html
  ADD requirements.txt requirements.txt
  ADD app.py app.py

  RUN pip3 install -r requirements.txt
  RUN nltkmodels.py

  EXPOSE 5000
  CMD [ "python3", "app.py" ]