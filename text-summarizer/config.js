
let settings = {
    scriptPath : '../../scripts',
    dataPath : './../scripts',
    changeScriptPath : function(newPath){
        this.scriptPath = newPath;
    },
    changeDataPath : function(newPath){
        this.dataPath = newPath;
    }
};

exports.settings = settings;

