from flask import Flask,jsonify,request
from classi import get_pred
app = Flask(__name__)
@app.route("/pred-alpha",methods = ["POST"])
def predict_data():
    image = request.files.get("digit")
    prediction=get_prediction(image)
    return jsonify({
        "your alphabet is":prediction
        
    }),200

if __name__ =="__main__":
    app.run(debug=True)
    