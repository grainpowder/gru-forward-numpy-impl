FROM python:3.7

ENV APP_PATH /opt/app
ENV PYTHONPATH $APP_PATH/src

COPY requirements.txt $APP_PATH/requirements.txt
WORKDIR $APP_PATH
RUN pip3 install -r requirements.txt

COPY src $APP_PATH/src
COPY data $APP_PATH/data

CMD /bin/bash
