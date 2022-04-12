from flask import Flask,request,jsonify
import joblib
import pandas as pd



app=Flask(__name__)

@app.route('/predict',methods=['POST'])

def predict():
    
    print('Request recieved....Loading model..... ')
    loaded_model=joblib.load('svm_model.pkl')
    vectorizer=joblib.load('vectorizer.pkl')
    print('Model Initialized!!!!')
    print('Processing Input file......')
    df=pd.read_csv(request.files.get('file'))
    df=df.drop(['ID','Thumbs Up','Developer Reply','Review Date','App ID','Version'],axis=1)
    df=df.dropna()
    review_list=[]
    for index,row in df.iterrows():
      review=[row['Text']]
      review=vectorizer.transform(review)
      result=loaded_model.predict(review)
      star=row['Star']
      url=row['Review URL']
      user_name=row['User Name']
      if(result=='positive' and star<3):
        review_dict={}
        review_dict['star']=str(star)
        review_dict['url']=url
        review_dict['user_name']=user_name
        review_dict['review']=row['Text']
        review_list.append(review_dict)
    print('File processing complete!!!')    
    return jsonify(results=review_list)


if __name__=='__main__':
    
    app.run(debug=True)