from flask import Flask,request,jsonify
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import pandas as pd


app=Flask(__name__)

@app.route('/predict',methods=['POST'])

def predict():
    
    print('Request recieved....Initializing model..... ')
    tokenizer=AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model=AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    print('Model Initialized!!!!')
    print('Processing Input file......')
    df=pd.read_csv(request.files.get('file'))
    df=df.drop(['ID','Thumbs Up','Developer Reply','Review Date','App ID','Version'],axis=1)
    df=df.dropna()
    review_list=[]
    for ind in df.index:
      review=df['Text'][ind]
      tokens=tokenizer.encode(review,return_tensors='pt')
      result=model(tokens)
      star=df['Star'][ind]
      url=df['Review URL'][ind]
      user_name=df['User Name'][ind]
      sentence_score=int(torch.argmax(result.logits))+1
      if(sentence_score-star>2):
        review_dict={}
        review_dict['star']=str(star)
        review_dict['url']=url
        review_dict['user_name']=user_name
        review_dict['review']=review
        review_list.append(review_dict)
    print('File processing complete!!!')    
    return jsonify(results=review_list)


if __name__=='__main__':
    
    app.run(debug=True)