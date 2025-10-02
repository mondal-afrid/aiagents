from flask import Flask, request, jsonify
from agent import AIAgent

app = Flask(__name__)
agent = AIAgent()

# preload some sample memory
agent.memory.add_episodic("User requested refund for a headset on Sept 20.")
agent.memory.add_episodic("User contacted support for a late delivery on Sept 10.")
agent.memory.add_semantic({"refund_policy": "Refunds allowed within 30 days of purchase."})
agent.memory.add_procedure("refund", ["Verify request", "Check policy", "Execute refund"])

@app.route('/interact', methods=['POST'])
def interact():
    data = request.json
    user_input = data.get("input", "")
    response = agent.process_request(user_input)
    return jsonify(response)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
