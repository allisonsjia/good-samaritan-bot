from flask import Flask, jsonify, render_template, request
from dispatcher import DispatcherApp  # Import the dispatcher response function
import nltk
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt_tab')

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