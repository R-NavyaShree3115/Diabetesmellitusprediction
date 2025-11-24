from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="../templates", static_folder="../static")

@app.route("/")
def home():
    return "Flask on Vercel Works!"

# Vercel handler
def handler(request, *args, **kwargs):
    return app(request.environ, start_response)

# Required for Vercel
from werkzeug.serving import run_simple

def start_response(status, headers):
    return None
