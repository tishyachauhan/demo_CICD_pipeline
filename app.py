import os
import json
import joblib
import numpy as np
from flask import Flask,request,jsonify

# config (defining variable "MODEL_PATH" where , the path of model is given as value to run easily)
MODEL_PATH = os.getenv("MODEL_PATH","model/iris_model.pkl")

# APP
app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading the model from {MODEL_PATH}: {e}")

@app.get('/health')
def health():
    return {"status": "ok"},200

# @app.get('/model_info')
# def model_info():
#     try:
#         model_info = {
#             "model_type": type(model).__name__,
#             "model_params": model.get_params()
#         }
#         input_format = {

#         }
#         return jsonify(model_info), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500



@app.post('/predict')
def predict():
    """accepts:
        {input:[[feature vector...]]}  #2d list
        OR
        {input:[feature vector...]}     #1d list
    """
    try:
        payload = request.json(force=True)
        x = payload.get("input")
        if x is None:
            return jsonify({"error": "Missing 'input' in request"}), 400
        # Normalise to 2d array
        if isinstance(x[0], list) and len(x[0])>0 and not isinstance(x[0], list):
            x = [x]

        X = np.array(x,dtype=float)
        preds = model.predict(X)

        # if model returns numpy types, convert to native python types for JSON serialization
        preds = preds.tolist()
        return jsonify({"predictions": preds}), 200
        # or return jsonify(prediction=preds),200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",5000)))
            

    

