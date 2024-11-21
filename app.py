from flask import Flask, jsonify, render_template, request
from dispatcher import DispatcherApp  # Import the dispatcher response function

app = Flask(__name__, static_folder='static')
DISPATCHER = DispatcherApp("7623f706-02e2-427e-8e10-c1b77db64b56", 
    "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")

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle user messages and return a dispatcher response."""
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    # Get a response from the dispatcher assistant
    dispatcher_response = DISPATCHER.query_app(user_message)
    return jsonify({"user_message": user_message, "dispatcher_response": dispatcher_response})

if __name__ == '__main__':
    app.run(debug=True)