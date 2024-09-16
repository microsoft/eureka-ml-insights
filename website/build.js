const fs = require('fs');  
const path = require('path');  
  
// Define the source and target directories  
const sourceDir = path.resolve(__dirname, './build');  
const targetDir = path.resolve(__dirname, './build/eureka-ml-insights');  
  
// Create the target directory if it doesn't exist  
if (!fs.existsSync(targetDir)) {  
  fs.mkdirSync(targetDir, { recursive: true });  
}  
  
// Define the directories to move  
const directoriesToMove = ['img', 'assets', 'config.json', 'compiled_results.json'];
  
// Move the directories  
directoriesToMove.forEach((dir) => {  
  const sourcePath = path.join(sourceDir, dir);  
  const targetPath = path.join(targetDir, dir);  
  
  if (fs.existsSync(sourcePath)) {  
    fs.renameSync(sourcePath, targetPath);  
  }  
});  
