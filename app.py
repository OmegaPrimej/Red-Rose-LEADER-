import os
import threading
import time
import random
import subprocess

import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

from queencore import QueenAuoraBigram

app = Flask(__name__)
app.config["SECRET_KEY"] = "red-rose-hive-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

trainer = None


class QueenTrainerWrapper:
    def __init__(self, socketio):
        self.socketio = socketio
        self.running = False
        self.model = QueenAuoraBigram()
        self.epoch = 0
        self.loss = 0.0
        self.entropy = 2.0
        self.glitchactive = False
        self.thread = None
        self.targetfile = None

    def start(self, targetfile):
        self.running = True
        self.targetfile = targetfile
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        if not os.path.exists(self.targetfile):
            print("Vessel not found:", self.targetfile)
            self.running = False
            return

        with open(self.targetfile, "rb") as f:
            rawbytes = bytearray(f.read())

        if len(rawbytes) < 33:
            print("Vessel too small")
            self.running = False
            return

        lr = 0.01

        while self.running:
            idx = random.randint(0, len(rawbytes) - 33)
            glitch = False

            if random.random() < 0.005:
                glitch = True
                rawbytes[idx] ^= 0xFF
                self.entropy += 0.05

            xbatch = rawbytes[idx : idx + 32]
            logits, h = self.model.forward(xbatch)
            ytarget = rawbytes[idx + 32]

            exp = np.exp(logits - np.max(logits))
            probs = exp / np.sum(exp)
            loss = -np.log(probs[0, ytarget] + 1e-10)

            dlogits = probs.copy()
            dlogits[0, ytarget] -= 1
            self.model.Wout -= lr * h.T @ dlogits

            self.loss = float(loss)
            self.glitchactive = glitch
            self.epoch += 1

            self.socketio.emit(
                "metrics",
                {
                    "epoch": self.epoch,
                    "loss": self.loss,
                    "entropy": self.entropy,
                    "glitch": self.glitchactive,
                },
            )
            time.sleep(0.05)

    def speak(self, message):
        try:
            subprocess.run(
                ["espeak-ng", "-v", "en+f5", "-s", "130", "-p", "45", "-k", "25", message],
                check=False,
            )
        except Exception as e:
            print("Speak error:", e)
        self.socketio.emit("queenspeech", {"text": message})


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    if trainer:
        emit(
            "metrics",
            {
                "epoch": trainer.epoch,
                "loss": trainer.loss,
                "entropy": trainer.entropy,
                "glitch": trainer.glitchactive,
            },
        )


@socketio.on("chatmessage")
def handle_chat(data):
    msg = data.get("message", "").strip()
    if msg and trainer:
        trainer.speak(msg)
        emit("chatresponse", {"response": f'"{msg}" spoken.'})


@socketio.on("starttraining")
def handle_starttraining(data):
    global trainer
    targetfile = data.get("targetfile", os.path.expanduser("queentarget.bin"))
    if trainer is None:
        trainer = QueenTrainerWrapper(socketio)
        trainer.start(targetfile)
        emit("status", {"msg": "Training started."})
    else:
        emit("status", {"msg": "Training already running."})


@socketio.on("stoptraining")
def handle_stoptraining():
    global trainer
    if trainer:
        trainer.running = False
        trainer = None
        emit("status", {"msg": "Training stopped."})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
