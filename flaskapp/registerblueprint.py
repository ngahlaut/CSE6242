from flask_restx import Api
from flask import Blueprint
import flaskapp.routes as product ## Do not move this to the top its here because of circular reference

def registerBluePrint(app):
    product_bp = Blueprint('product', __name__, url_prefix='/swagger')
    product_api = Api(product_bp, title='COVID19 KnowledgeGraph', description='API to build COVID19 Knowledge Grapch')
    product_api.add_namespace(product.product_ns)
    app.register_blueprint(product_bp)