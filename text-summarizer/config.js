
let settings = {
    scriptPath : '../scripts',
    dataPath : '.',
    changeScriptPath : function(newPath){
        this.scriptPath = newPath;
    },
    changeDataPath : function(newPath){
        this.dataPath = newPath;
    }
};

exports.settings = settings;

