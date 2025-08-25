import dotenv from 'dotenv';
dotenv.config();

export const CONFIG = {
  PORT: process.env.PORT || 3000,
  NODE_ENV: process.env.NODE_ENV || 'development',
  MONGODB_URI: process.env.MONGODB_URI,
  JWT_SECRET: process.env.JWT_SECRET,
  JWT_EXPIRES: process.env.JWT_EXPIRES || '7d',
  GOOGLE_MAPS_API_KEY: process.env.GOOGLE_MAPS_API_KEY,
  UPLOAD_DIR: new URL('../../uploads', import.meta.url).pathname,
  PUBLIC_DIR: new URL('../../public', import.meta.url).pathname,
};
