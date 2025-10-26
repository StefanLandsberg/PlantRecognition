import mongoose from 'mongoose';
const schema = new mongoose.Schema({
  username: {type: String, unique: true, required: true, index: true},
  email: { type: String, unique: true, required: true },
  passwordHash: { type: String, required: true },
  role: { type: String, enum: ['user','admin'], default: 'user' },
  storagePreference: {
    type: String,
    enum: ['server', 'local'],
    default: 'server',
    description: 'User preference for file storage: server (2GB limit, 90 days retention) or local (unlimited)'
  }
}, { timestamps: true });
export default mongoose.model('User', schema);
