// Modules to control application life and create native browser window
const {app, BrowserWindow, Menu, ipcMain} = require('electron')
const {createSettingsWindow, listenForSettingsCancel, listenForSettingsChange} = require('./settings');
const {listenForConvTrigger, listenForRecurrentTrigger, listenForLSATrigger} = require('./linkers/modelListeners');

// Production mode
process.env.NODE_ENV = 'production'


let mainWindow

function createWindow () {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    height: 800,
    width: 1200,
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
  
  // Model triggers and responses
  ipcMain.on('trigger:conv', function (){
    listenForConvTrigger(mainWindow)
  });
  ipcMain.on('trigger:rec', function (){
    listenForRecurrentTrigger(mainWindow)
  });
  ipcMain.on('trigger:lsa', function (){
    listenForLSATrigger(mainWindow)
  });
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
listenForSettingsCancel();
listenForSettingsChange();