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