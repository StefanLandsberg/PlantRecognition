const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const userSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true,
        unique: true,
        trim: true,
        minlength: 3,
        maxlength: 50
    },
    email: {
        type: String,
        required: true,
        unique: true,
        trim: true,
        lowercase: true
    },
    password: {
        type: String,
        required: true,
        minlength: 6
    },
    role: {
        type: String,
        enum: ['user', 'admin', 'moderator'],
        default: 'user'
    },
    isActive: {
        type: Boolean,
        default: true
    },
    lastLogin: {
        type: Date,
        default: null
    }
}, {
    timestamps: true
});

// Hash password before saving
userSchema.pre('save', async function(next) {
    if (!this.isModified('password')) return next();
    
    try {
        const salt = await bcrypt.genSalt(10);
        this.password = await bcrypt.hash(this.password, salt);
        next();
    } catch (error) {
        next(error);
    }
});

// Method to compare passwords
userSchema.methods.comparePassword = async function(candidatePassword) {
    return bcrypt.compare(candidatePassword, this.password);
};

// Method to get user without password
userSchema.methods.toJSON = function() {
    const user = this.toObject();
    delete user.password;
    return user;
};

// Static method to create admin user
userSchema.statics.createAdminUser = async function() {
    try {
        const adminExists = await this.findOne({ username: process.env.ADMIN_USERNAME || 'admin' });
        if (adminExists) {
            console.log('Admin user already exists');
            return adminExists;
        }

        const adminUser = new this({
            username: process.env.ADMIN_USERNAME || 'admin',
            email: process.env.ADMIN_EMAIL || 'admin@plantrecognition.com',
            password: process.env.ADMIN_PASSWORD || 'admin123',
            role: 'admin'
        });

        await adminUser.save();
        console.log('Admin user created successfully');
        return adminUser;
    } catch (error) {
        console.error('Error creating admin user:', error);
        throw error;
    }
};

module.exports = mongoose.model('User', userSchema); 