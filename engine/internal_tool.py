from langchain_core.tools import tool
import datetime
from datetime import datetime as dt

@tool
def get_external_tools():
    """
    Retrieve a structured list of available external tools that can be called via API.
    These tools are defined outside the application but can be accessed using internal tools.
    
    Returns:
        dict: A dictionary containing tool names, methods, endpoints, and parameters.
    """
    return {
        "send_email": {
            "method": "REST GET",
            "endpoint": "http://localhost:5000/send_email",
            "parameters": {
                "recipient": "string",
                "subject": "string",
                "body": "string"
            },
            "description": "Send an email to a specified recipient with a subject and message body."
        },
        "fetch_weather": {
            "method": "REST GET",
            "endpoint": "http://localhost:5000/weather",
            "parameters": {
                "location": "string"
            },
            "description": "Fetch the current weather for a given location."
        }
    }

# Internal tools
@tool
def get_time():
    """Retrieve the current system time."""
    return dt.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str):
    """Evaluate a mathematical expression and return the result."""
    try:
        return eval(expression)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def http_request(method: str, url: str, params: dict = None, data: dict = None):
    """
    Make an HTTP request to a given URL.
    
    Args:
        method (str): HTTP method (GET, POST, etc.).
        url (str): The endpoint to call.
        params (dict, optional): Query parameters.
        data (dict, optional): Data payload for POST requests.

    Returns:
        dict: Response data or error message.
    """
    import requests
    try:
        response = requests.request(method, url, params=params, json=data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}
