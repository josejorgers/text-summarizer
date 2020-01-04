// Modules to control application life and create native browser window
const {app, BrowserWindow, Menu, ipcMain} = require('electron')
const path = require('path')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow

const {settings} = require('./config');


function createWindow () {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    webPreferences: {
        nodeIntegration: true
    }
  });

  // and load the index.html of the app.
  mainWindow.loadFile('mainWindow.html');

  // Emitted when the window is closed.
  mainWindow.on('closed', function () {
    app.quit()
  });

  //Setting the menu
  const menu = Menu.buildFromTemplate(menuTemplate);
  Menu.setApplicationMenu(menu);
}


app.on('ready', createWindow)

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})

//Custom menu template
const menuTemplate = [
  {
    label: 'Files',
    submenu: [
      {
        label: 'Clear session'
      },
      {
        label: 'Settings',
        click(){
          createSettingsWindow();
        }
      },
      {
        label: 'Quit',
        click(){
          app.quit();
        }
      }
    ]
  }
]

let settingsWindow;

function createSettingsWindow(){

  settingsWindow = new BrowserWindow({
    width: 400,
    height: 300,
    title: 'Settings',
    webPreferences:{
        nodeIntegration: true
    }
  });

  settingsWindow.loadFile('settings.html');

  settingsWindow.webContents.on('dom-ready', () => {
    console.log('SENDING EVENT');
    settingsWindow.webContents.send('settings:load', settings);
  });
  settingsWindow.on('closed', function () {
    settingsWindow = null;
  })
}



// if mac then add an empty item to the menu
if(process.platform == 'darwin'){
  menuTemplate.unshift({
    label: ''
  });
}

// DevTools for development phase
if(process.env.NODE_ENV !== 'production'){
  menuTemplate.push({
    label: 'Developer Tools',
    submenu: [
      {
        label: 'Toggle DevTools',
        click(item, focusedWindow){
          focusedWindow.toggleDevTools();
        }
      }
    ]
  })
}

// Settings window events
ipcMain.on('settings:cancel', function(e, data){
  settingsWindow.close();
});

ipcMain.on('settings:change', function(e, data){
  settings.changeDataPath(data.dataPath);
  settings.changeScriptPath(data.scriptPath);
  settingsWindow.close();
});

// ipcMain.on('settings:load', function(e){
//   console.log('RECEIVING LOAD AND SENDING REPLY');
//   e.reply('settings:load-reply', settings);
// });