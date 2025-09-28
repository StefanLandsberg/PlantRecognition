import { execSync } from 'child_process';
import fs from 'fs';

try {
  console.log('Generating SSL certificates...');

  const command = 'openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=ZA/ST=GP/L=JHB/O=PlantApp/CN=192.168.101.251"';

  execSync(command, { stdio: 'inherit' });

  // Check if files were created
  if (fs.existsSync('key.pem') && fs.existsSync('cert.pem')) {
    console.log('SSL certificates generated successfully!');
    console.log('Files created: key.pem, cert.pem');
  } else {
    console.log('Certificate generation failed');
  }
} catch (error) {
  console.error('Error generating certificates:', error.message);
}