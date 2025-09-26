#!/usr/bin/env python3

from flask import Flask

app = Flask(__name__)

@app.route('/api/test', methods=['GET'])
def test():
    return {'status': 'working', 'message': 'Test server is running!'}

if __name__ == '__main__':
    print("Starting test server...")
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)