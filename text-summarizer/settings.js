const {BrowserWindow, ipcMain} = require('electron');
const {settings} = require('./config');
const path = require('path');

let settingsWindow;

function listenForSettingsCancel(){
    ipcMain.on('settings:cancel', function(e, data){
        settingsWindow.close();
    });
}
  
function listenForSettingsChange(){
    ipcMain.on('settings:change', function(e, data){
      settings.changeDataPath(path.join(__dirname, data.dataPath));
      settings.changeScriptPath(path.join(__dirname, data.scriptPath));
      settingsWindow.close();
    });
}


function createSettingsWindow(){

    settingsWindow = new BrowserWindow({
      width: 800,
      height: 400,
      title: 'Settings',
      webPreferences:{
          nodeIntegration: true
      }
    });
  
    settingsWindow.loadFile('settings.html');
  
    settingsWindow.webContents.on('dom-ready', () => {
      settingsWindow.webContents.send('settings:load', settings);
    });
    settingsWindow.on('closed', function () {
      settingsWindow = null;
    })
  }

exports.settingsWindow = settingsWindow;
exports.createSettingsWindow = createSettingsWindow;
exports.listenForSettingsCancel = listenForSettingsCancel;
exports.listenForSettingsChange = listenForSettingsChange;