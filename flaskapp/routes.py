from flask.wrappers import Response
from flask_restx import Namespace, Resource
import pandas as pd
from flask import render_template,request, jsonify
import flaskapp.queryhandler as qh
import flaskapp
import os

product_ns = Namespace('COVID-19 Knowledge Graph', description='COVID-19 Knowledge Graph')

@flaskapp.flask_app.route('/')
@flaskapp.flask_app.route('/default')
@flaskapp.flask_app.route('/home')
def home():
    #os.chdir(os.path.dirname(__file__))
    return render_template('submission.html')

@product_ns.route('/getqueryembedding')
@product_ns.doc("API that returns a query embedding for a given query string")
@product_ns.doc(params={'querystr': 'Query String'})
class queryembedding(Resource):
    def get(self):
        query_string = request.args.get("querystr")
        embedding = qh.getqueryembedding(query_string,query_string)
        embedding = pd.DataFrame(embedding).astype("float")
        flaskapp.flask_app.logger.info("getqueryembedding Success")
        return embedding.to_json(orient="records")

@product_ns.route('/getrelevanttitles')
@product_ns.doc("API that returns relevant titles for a query  string")
@product_ns.doc(params={'querystr': 'Query String'})
class relateddocuments(Resource):
    def get(self):
        query_string = request.args.get("querystr")
        flaskapp.flask_app.logger.info("getrelevanttitles Success")
        return qh.getrelateddocuments(query_string,query_string)


