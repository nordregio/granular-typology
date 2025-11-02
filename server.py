from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route("/<path:filename>")
def serve_static_file(filename):
    return send_from_directory(".", filename)

if __name__ == "__main__":
    app.run(debug=True)
