const { app, BrowserWindow } = require("electron");

function createWindow() {
    const window = new BrowserWindow({
        width: 1600,
        height: 900,
    });

    window.maximize();

    window.loadFile("index.html");
}

app.whenReady().then(createWindow);