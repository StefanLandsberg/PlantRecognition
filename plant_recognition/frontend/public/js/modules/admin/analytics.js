// Analytics Tab Manager
import { api } from '../api.js';
import { handleError } from '../utils.js';

class AnalyticsManager {
    constructor() {
        this.sightingsData = [];
        this.charts = {};
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        try {
            console.log('Initializing Analytics Manager...');
            await this.loadSightingsData();
            this.initialized = true;
            console.log('Analytics Manager initialized successfully');
        } catch (error) {
            console.error('Error initializing Analytics Manager:', error);
            handleError(error, 'analytics_initialization');
        }
    }

    async loadSightingsData() {
        try {
            const response = await api.getSightings();
            if (response.success && response.data) {
                this.sightingsData = response.data;
                console.log(`Loaded ${this.sightingsData.length} sightings for analytics`);
            }
        } catch (error) {
            console.error('Error loading sightings data for analytics:', error);
        }
    }

    async refreshData() {
        await this.loadSightingsData();
        if (this.charts.speciesBreakdown) {
            this.updateAnalyticsCharts();
        }
    }

    onTabActivated() {
        this.updateAnalyticsCharts();
    }

    updateAnalyticsCharts() {
        this.createSpeciesBreakdownChart();
        this.createSpreadTrendsChart();
        this.createDetectionFrequencyChart();
        this.createMonthlyGrowthChart();
    }

    createSpeciesBreakdownChart() {
        const ctx = document.getElementById('species-breakdown-chart');
        if (!ctx) {
            console.warn('Species breakdown chart container not found');
            return;
        }

        try {
            // Destroy existing chart if it exists
            if (this.charts.speciesBreakdown) {
                this.charts.speciesBreakdown.destroy();
            }

            const speciesData = this.getSpeciesBreakdownData();
            
            if (speciesData.labels.length === 0) {
                ctx.parentElement.innerHTML = '<div class="chart-placeholder">No data available for species breakdown</div>';
                return;
            }
            
            this.charts.speciesBreakdown = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: speciesData.labels,
                    datasets: [{
                        data: speciesData.values,
                        backgroundColor: this.generateColors(speciesData.labels.length),
                        borderWidth: 2,
                        borderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 20,
                                usePointStyle: true,
                                color: '#ffffff'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Species Detection Breakdown',
                            font: {
                                size: 16,
                                weight: 'bold'
                            },
                            color: '#ffffff'
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating species breakdown chart:', error);
            if (ctx.parentElement) {
                ctx.parentElement.innerHTML = '<div class="chart-placeholder">Error loading chart</div>';
            }
        }
    }

    createSpreadTrendsChart() {
        const ctx = document.getElementById('spread-trends-chart');
        if (!ctx) {
            console.warn('Spread trends chart container not found');
            return;
        }

        try {
            if (this.charts.spreadTrends) {
                this.charts.spreadTrends.destroy();
            }

            const trendsData = this.getSpreadTrendsData();
            
            if (trendsData.labels.length === 0) {
                ctx.parentElement.innerHTML = '<div class="chart-placeholder">No data available for spread trends</div>';
                return;
            }
            
            this.charts.spreadTrends = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: trendsData.labels,
                    datasets: [{
                        label: 'Invasive Species',
                        data: trendsData.invasive,
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        tension: 0.4,
                        fill: true
                    }, {
                        label: 'Native Species',
                        data: trendsData.native,
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                color: '#ffffff'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Species Spread Trends (Last 6 Months)',
                            font: {
                                size: 16,
                                weight: 'bold'
                            },
                            color: '#ffffff'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Detections',
                                color: '#ffffff'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#ffffff'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Month',
                                color: '#ffffff'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#ffffff'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating spread trends chart:', error);
            if (ctx.parentElement) {
                ctx.parentElement.innerHTML = '<div class="chart-placeholder">Error loading chart</div>';
            }
        }
    }

    createDetectionFrequencyChart() {
        const ctx = document.getElementById('detection-frequency-chart');
        if (!ctx) {
            console.warn('Detection frequency chart container not found');
            return;
        }

        try {
            if (this.charts.detectionFrequency) {
                this.charts.detectionFrequency.destroy();
            }

            const frequencyData = this.getDetectionFrequencyData();
            
            if (frequencyData.labels.length === 0) {
                ctx.parentElement.innerHTML = '<div class="chart-placeholder">No data available for detection frequency</div>';
                return;
            }
            
            this.charts.detectionFrequency = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: frequencyData.labels,
                    datasets: [{
                        label: 'Detections per Day',
                        data: frequencyData.values,
                        backgroundColor: 'rgba(54, 162, 235, 0.8)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Daily Detection Frequency (Last 30 Days)',
                            font: {
                                size: 16,
                                weight: 'bold'
                            },
                            color: '#ffffff'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Detections',
                                color: '#ffffff'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#ffffff'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Date',
                                color: '#ffffff'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#ffffff'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating detection frequency chart:', error);
            if (ctx.parentElement) {
                ctx.parentElement.innerHTML = '<div class="chart-placeholder">Error loading chart</div>';
            }
        }
    }

    createMonthlyGrowthChart() {
        const ctx = document.getElementById('monthly-growth-chart');
        if (!ctx) {
            console.warn('Monthly growth chart container not found');
            return;
        }

        try {
            if (this.charts.monthlyGrowth) {
                this.charts.monthlyGrowth.destroy();
            }

            const growthData = this.getMonthlyGrowthData();
            
            if (growthData.labels.length === 0) {
                ctx.parentElement.innerHTML = '<div class="chart-placeholder">No data available for monthly growth</div>';
                return;
            }
            
            this.charts.monthlyGrowth = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: growthData.labels,
                    datasets: [{
                        label: 'Total Detections',
                        data: growthData.values,
                        borderColor: '#6f42c1',
                        backgroundColor: 'rgba(111, 66, 193, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointBackgroundColor: '#6f42c1',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                color: '#ffffff'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Monthly Growth in Detections',
                            font: {
                                size: 16,
                                weight: 'bold'
                            },
                            color: '#ffffff'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Total Detections',
                                color: '#ffffff'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#ffffff'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Month',
                                color: '#ffffff'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#ffffff'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating monthly growth chart:', error);
            if (ctx.parentElement) {
                ctx.parentElement.innerHTML = '<div class="chart-placeholder">Error loading chart</div>';
            }
        }
    }

    getSpeciesBreakdownData() {
        const speciesCount = {};
        
        this.sightingsData.forEach(sighting => {
            const species = sighting.species;
            speciesCount[species] = (speciesCount[species] || 0) + 1;
        });

        const sortedSpecies = Object.entries(speciesCount)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10); // Top 10 species

        return {
            labels: sortedSpecies.map(([species]) => species),
            values: sortedSpecies.map(([, count]) => count)
        };
    }

    getSpreadTrendsData() {
        const months = [];
        const invasiveData = [];
        const nativeData = [];
        
        // Generate last 6 months
        for (let i = 5; i >= 0; i--) {
            const date = new Date();
            date.setMonth(date.getMonth() - i);
            months.push(date.toLocaleDateString('en-US', { month: 'short' }));
            
            const monthStart = new Date(date.getFullYear(), date.getMonth(), 1);
            const monthEnd = new Date(date.getFullYear(), date.getMonth() + 1, 0);
            
            const monthSightings = this.sightingsData.filter(sighting => {
                const sightingDate = new Date(sighting.timestamp);
                return sightingDate >= monthStart && sightingDate <= monthEnd;
            });
            
            const invasive = monthSightings.filter(s => s.llmAnalysis?.invasive_status === true).length;
            const native = monthSightings.filter(s => s.llmAnalysis?.invasive_status === false).length;
            
            invasiveData.push(invasive);
            nativeData.push(native);
        }

        return {
            labels: months,
            invasive: invasiveData,
            native: nativeData
        };
    }

    getDetectionFrequencyData() {
        const dailyCounts = {};
        
        // Generate last 30 days
        for (let i = 29; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            dailyCounts[dateStr] = 0;
        }
        
        // Count detections per day
        this.sightingsData.forEach(sighting => {
            const sightingDate = new Date(sighting.timestamp);
            const dateStr = sightingDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            if (dailyCounts.hasOwnProperty(dateStr)) {
                dailyCounts[dateStr]++;
            }
        });

        return {
            labels: Object.keys(dailyCounts),
            values: Object.values(dailyCounts)
        };
    }

    getMonthlyGrowthData() {
        const monthlyTotals = {};
        
        this.sightingsData.forEach(sighting => {
            const sightingDate = new Date(sighting.timestamp);
            const monthKey = sightingDate.toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
            monthlyTotals[monthKey] = (monthlyTotals[monthKey] || 0) + 1;
        });

        const sortedMonths = Object.keys(monthlyTotals).sort();
        
        return {
            labels: sortedMonths,
            values: sortedMonths.map(month => monthlyTotals[month])
        };
    }

    generateColors(count) {
        const colors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
            '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384'
        ];
        
        const result = [];
        for (let i = 0; i < count; i++) {
            result.push(colors[i % colors.length]);
        }
        return result;
    }
}

// Create singleton instance
export const analyticsManager = new AnalyticsManager(); 