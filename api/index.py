from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="../templates", static_folder="../static")

@app.route("/")
def home():
    return "Flask on Vercel!"

def handler(event, context):
    return app(event, context)
