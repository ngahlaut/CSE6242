from flaskapp import flask_app
from flaskapp.registerblueprint import registerBluePrint

registerBluePrint(flask_app)
flask_app.run(port=8080,debug=True)
