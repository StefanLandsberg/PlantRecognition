const express = require('express');
const router = express.Router();
const sightingController = require('../controllers/sightingController');

// Dashboard statistics
router.get('/stats', sightingController.getDashboardStats);

// Analytics endpoints
router.get('/analytics/species-breakdown', sightingController.getSpeciesBreakdown);
router.get('/analytics/spread-trends', sightingController.getSpreadTrends);
router.get('/analytics/frequency-data', sightingController.getFrequencyData);
router.get('/analytics/growth-data', sightingController.getGrowthData);

// Map data
router.get('/map', sightingController.getMapData);

// CRUD operations
router.post('/', sightingController.createSighting);
router.get('/', sightingController.getSightings);
router.get('/:id', sightingController.getSighting);
router.put('/:id', sightingController.updateSighting);

// LLM analysis update
router.post('/update-llm', sightingController.updateLLMAnalysis);

// Export functionality - fixed route to match frontend
router.get('/export', sightingController.exportSightings);

module.exports = router; 