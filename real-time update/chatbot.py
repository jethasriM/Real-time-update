import aiml
from flask import Flask, request, jsonify

app = Flask(__name__)
kernel = aiml.Kernel()
kernel.learn("availability.aiml")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = kernel.respond(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000)

