"""
Port 8051 to 8501 HTTP Redirector.
Ensures visits to http://localhost:8051 (e.g. /Animal_Records) automatically
route to the Streamlit Dashboard on port 8501.
"""

import http.server
import socketserver
import sys

SRC_PORT = 8051
TARGET_PORT = 8501


class RedirectHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        target_url = f"http://localhost:{TARGET_PORT}{self.path}"
        self.send_response(302)
        self.send_header("Location", target_url)
        self.end_headers()

    def do_HEAD(self):
        target_url = f"http://localhost:{TARGET_PORT}{self.path}"
        self.send_response(302)
        self.send_header("Location", target_url)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Quiet logging


def run_redirector():
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", SRC_PORT), RedirectHandler) as httpd:
            print(f"Port {SRC_PORT} redirector active -> http://localhost:{TARGET_PORT}")
            httpd.serve_forever()
    except Exception as e:
        print(f"Notice on port {SRC_PORT}: {e}", file=sys.stderr)


if __name__ == "__main__":
    run_redirector()
