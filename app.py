from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# Use environment variable for Hugging Face token (important for security)
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set this in Render's environment variables

# Initialize the Hugging Face InferenceClient
client = InferenceClient(model="stabilityai/stable-diffusion-3.5-large", token=HF_TOKEN)

# Route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prompt and fetch generated design
@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get the prompt from the frontend
        prompt = request.json.get("prompt")

        if not prompt:
            return jsonify({"error": "No prompt provided."}), 400

        # Log the prompt for debugging
        print(f"Prompt received: {prompt}")

        # Path to the image file
        image_path = "static/generated_image.png"

        # Delete the previous image if it exists
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Previous image deleted: {image_path}")

        # Use Hugging Face InferenceClient to generate an image from the prompt
        image = client.text_to_image(prompt)

        # Save the new image to the 'static' folder
        image.save(image_path)
        print(f"New image saved: {image_path}")

        return jsonify({"image_url": image_path})

    except Exception as e:
        # Log the unexpected error for debugging
        print(f"Unexpected Error: {str(e)}")
        return jsonify({"error": "Something went wrong with the API call."}), 500


if __name__ == '__main__':
    # Make sure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    # Fix Render.com port binding issue
    port = int(os.environ.get("PORT", 5000))  # Render assigns a dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)
