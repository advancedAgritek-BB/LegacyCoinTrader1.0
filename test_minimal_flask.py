#!/usr/bin/env python3
"""
Minimal Flask test to check if Flask is working correctly.
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello from minimal Flask!"

@app.route('/test')
def test():
    return {"message": "Flask is working!", "status": "success"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=True)
