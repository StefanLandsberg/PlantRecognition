// server/services/ml.service.js
import path from 'path';
import fs from 'fs';

function urlPathToFs(fileUrlPath) {
  const rel = String(fileUrlPath || '').replace(/^\/+/, '');
  const uploadsPath = path.join(process.cwd(), '..', 'uploads', path.basename(rel));

  if (fs.existsSync(uploadsPath)) {
    return uploadsPath;
  }

  const serverUploads = path.join(process.cwd(), rel);
  return serverUploads;
}

// Load class names from file
const classNamesPath = path.join(process.cwd(), '..', 'models', 'class_names.txt');
let classNames = ['Acacia_mearnsii', 'Acacia_melanoxylon', 'Acacia_podalyriifolia', 'Unknown_Plant'];

if (fs.existsSync(classNamesPath)) {
  try {
    const classNamesContent = fs.readFileSync(classNamesPath, 'utf-8');
    classNames = classNamesContent.split('\n').map(line => line.trim()).filter(line => line);
  } catch (e) {
    console.log('Could not load class names, using defaults');
  }
}

// Simple mock classifier - no spawn needed
export function runML(fileUrlPath) {
  const imagePathFs = urlPathToFs(fileUrlPath);

  return new Promise((resolve) => {
    console.log(`Processing image: ${imagePathFs}`);
    
    // Simple mock classification based on filename or random
    const randomIndex = Math.floor(Math.random() * classNames.length);
    const randomSpecies = classNames[randomIndex];
    const randomConfidence = 0.7 + Math.random() * 0.25; // 70-95% confidence
    
    // Simulate processing time
    setTimeout(() => {
      console.log(`Classification: ${randomSpecies} (${randomConfidence.toFixed(2)})`);
      
      resolve({
        predicted_species: randomSpecies,
        confidence: randomConfidence,
        raw: {
          predicted_species: randomSpecies,
          confidence: randomConfidence,
          top5_predictions: [
            [randomSpecies, randomConfidence],
            [classNames[(randomIndex + 1) % classNames.length], randomConfidence - 0.2],
            [classNames[(randomIndex + 2) % classNames.length], randomConfidence - 0.3]
          ]
        }
      });
    }, 500); // Half second delay to simulate processing
  });
}

