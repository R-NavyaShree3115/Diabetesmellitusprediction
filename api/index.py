from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="../templates", static_folder="../static")


@app.route("/")
def home():
    return "Flask on Render Works!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    return jsonify({"received": data, "status": "ok"})


# Local run support
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
