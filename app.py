from flask import Flask, render_template, request, send_from_directory
from gtts import gTTS
import os, uuid

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    output_text = ""
    audio_file = None

    if request.method == "POST":
        input_text = request.form.get("input_text")
        if input_text:
            output_text = input_text  # dummy output = same input

            # create unique mp3 filename
            filename = f"output_{uuid.uuid4().hex}.mp3"
            filepath = os.path.join("static", filename)

            # generate TTS
            tts = gTTS(text=output_text, lang="en")
            tts.save(filepath)

            audio_file = filename

    return render_template("index.html", output_text=output_text, audio_file=audio_file)

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
