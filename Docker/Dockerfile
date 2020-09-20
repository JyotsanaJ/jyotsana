FROM python:3.6
  
RUN pip3 install --no-cache-dir flask
RUN pip3 install fasttext
RUN pip3 install nltk
RUN pip3 install bs4
RUN pip3 install pandas
RUN pip3 install pickle-mixin
RUN pip3 install sklearn

WORKDIR /root

COPY . .

EXPOSE 9010

CMD ["python3", "/root/app.py"]
