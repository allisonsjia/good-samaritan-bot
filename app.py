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
    dispatcher_response = DISPATCHER.query_app(user_message, do_eval=False, with_planning=True)
    return jsonify({"user_message": user_message, "dispatcher_response": dispatcher_response})

if __name__ == '__main__':
    app.run(debug=True)