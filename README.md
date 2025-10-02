# Long-Term Memory AI Agent Demo

This project showcases an AI agent implemented in Python that simulates **episodic**, **semantic**, and **procedural** memory systems. The agent can process and recall information across different types of memory, demonstrating advanced memory management capabilities.

## Getting Started

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Start the AI agent server:**
    ```bash
    python app.py
    ```

3. **Interact with the agent using a `curl` command:**
    ```bash
    curl -X POST http://localhost:5000/interact -H "Content-Type: application/json" -d '{"your":"data"}'
    ```

Replace `/your-endpoint` and the JSON data with the appropriate values for your use case.

## Project Structure

- `app.py`: Main application entry point for the AI agent server.
- `requirements.txt`: Python dependencies.

## Notes

- Ensure the server is running before sending requests with `curl`.
- Adjust the `curl` command to test custom scenarios.
