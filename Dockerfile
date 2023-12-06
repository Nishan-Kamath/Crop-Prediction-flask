FROM python:3.8-slim

WORKDIR /app

ADD . /app

RUN pip install flask
RUN pip install Flask
RUN pip install requests
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install numpy

copy . .

CMD [ "python" , "-m" ,"flask" ,"run" ,"--host=0.0.0.0"]