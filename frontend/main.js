const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");

function createWindow() {

    const window = new BrowserWindow({
        width: 1600,
        height: 900,
    });

    window.maximize();

    const python = spawn(
        "python",
        ["../backend/main.py"],
        {
            cwd: __dirname
        }
    );

    python.stdout.on("data", (data) => {
        console.log(`Python: ${data}`);
    });

    python.stderr.on("data", (data) => {
        console.error(`Python error: ${data}`);
    });

    window.loadFile("index.html");
}

app.whenReady().then(createWindow);