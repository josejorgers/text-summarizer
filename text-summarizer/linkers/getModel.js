const {PythonShell} = require('python-shell');
const path = require('path');
const {settings} = require('../config');

function getModelResults(window, model){
    
    let callingArgs = {
        scriptPath : path.join(__dirname, settings.scriptPath),
        args:[
            model,
            path.join('', settings.dataPath)
        ]
    }

    let convResults = new PythonShell('inference.py', callingArgs);
    
    let event = model+':results';
    convResults.on('message', function(msg){
        window.webContents.send(event, msg);
    });

    //// Checking script errors (mostly dir path related)
    // convResults.on('stderr', function(msg){
    //     console.log(msg);
    // });
}

exports.getModelResults = getModelResults