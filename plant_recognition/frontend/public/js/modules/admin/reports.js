// Reports Tab Manager
import { api } from '../api.js';
import { sanitizeHtml } from '../../utils/sanitize.js';
import { handleError } from '../utils.js';

class ReportsManager {
    constructor() {
        this.sightingsData = [];
        this.currentFilters = {};
        this.filteredData = [];
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        try {
            console.log('Initializing Reports Manager...');
            await this.loadSightingsData();
            this.initialized = true;
            console.log('Reports Manager initialized successfully');
        } catch (error) {
            console.error('Error initializing Reports Manager:', error);
            handleError(error, 'reports_initialization');
        }
    }

    async loadSightingsData() {
        try {
            const response = await api.getSightings();
            if (response.success && response.data) {
                this.sightingsData = response.data;
                this.filteredData = [...this.sightingsData];
                console.log(`Loaded ${this.sightingsData.length} sightings for reports`);
            }
        } catch (error) {
            console.error('Error loading sightings data for reports:', error);
        }
    }

    async refreshData() {
        await this.loadSightingsData();
    }

    onTabActivated() {
        this.setupReports();
    }

    setupReports() {
        this.setupFilters();
        this.updateReportsTable();
        this.updateReportStats();
    }

    setupFilters() {
        // Date range filter
        const dateFromInput = document.getElementById('date-from');
        const dateToInput = document.getElementById('date-to');
        
        if (dateFromInput) {
            dateFromInput.addEventListener('change', () => this.applyFilters());
        }
        if (dateToInput) {
            dateToInput.addEventListener('change', () => this.applyFilters());
        }

        // Species filter
        const speciesFilter = document.getElementById('species-filter');
        if (speciesFilter) {
            this.populateSpeciesFilter();
            speciesFilter.addEventListener('change', () => this.applyFilters());
        }

        // Status filter
        const statusFilter = document.getElementById('status-filter');
        if (statusFilter) {
            statusFilter.addEventListener('change', () => this.applyFilters());
        }

        // Risk level filter
        const riskFilter = document.getElementById('risk-filter');
        if (riskFilter) {
            riskFilter.addEventListener('change', () => this.applyFilters());
        }

        // Clear filters button
        const clearFiltersBtn = document.getElementById('clear-filters');
        if (clearFiltersBtn) {
            clearFiltersBtn.addEventListener('click', () => this.clearFilters());
        }

        // Export buttons
        const exportCSVBtn = document.getElementById('export-csv');
        const exportPDFBtn = document.getElementById('export-pdf');
        
        if (exportCSVBtn) {
            exportCSVBtn.addEventListener('click', () => this.exportToCSV());
        }
        if (exportPDFBtn) {
            exportPDFBtn.addEventListener('click', () => this.exportToPDF());
        }
    }

    populateSpeciesFilter() {
        const speciesFilter = document.getElementById('species-filter');
        if (!speciesFilter) return;

        const uniqueSpecies = [...new Set(this.sightingsData.map(s => s.species))];
        
        speciesFilter.innerHTML = '<option value="">All Species</option>';
        uniqueSpecies.forEach(species => {
            const option = document.createElement('option');
            option.value = species;
            option.textContent = species;
            speciesFilter.appendChild(option);
        });
    }

    applyFilters() {
        this.currentFilters = {
            dateFrom: document.getElementById('date-from')?.value,
            dateTo: document.getElementById('date-to')?.value,
            species: document.getElementById('species-filter')?.value,
            status: document.getElementById('status-filter')?.value,
            riskLevel: document.getElementById('risk-filter')?.value
        };

        this.filteredData = this.sightingsData.filter(sighting => {
            // Date filter
            if (this.currentFilters.dateFrom || this.currentFilters.dateTo) {
                const sightingDate = new Date(sighting.timestamp);
                const fromDate = this.currentFilters.dateFrom ? new Date(this.currentFilters.dateFrom) : null;
                const toDate = this.currentFilters.dateTo ? new Date(this.currentFilters.dateTo) : null;
                
                if (fromDate && sightingDate < fromDate) return false;
                if (toDate && sightingDate > toDate) return false;
            }

            // Species filter
            if (this.currentFilters.species && sighting.species !== this.currentFilters.species) {
                return false;
            }

            // Status filter
            if (this.currentFilters.status && sighting.management?.status !== this.currentFilters.status) {
                return false;
            }

            // Risk level filter
            if (this.currentFilters.riskLevel && sighting.llmAnalysis?.risk_level !== this.currentFilters.riskLevel) {
                return false;
            }

            return true;
        });

        this.updateReportsTable();
        this.updateReportStats();
    }

    clearFilters() {
        // Reset filter inputs
        const dateFromInput = document.getElementById('date-from');
        const dateToInput = document.getElementById('date-to');
        const speciesFilter = document.getElementById('species-filter');
        const statusFilter = document.getElementById('status-filter');
        const riskFilter = document.getElementById('risk-filter');

        if (dateFromInput) dateFromInput.value = '';
        if (dateToInput) dateToInput.value = '';
        if (speciesFilter) speciesFilter.value = '';
        if (statusFilter) statusFilter.value = '';
        if (riskFilter) riskFilter.value = '';

        // Reset filters and data
        this.currentFilters = {};
        this.filteredData = [...this.sightingsData];
        
        this.updateReportsTable();
        this.updateReportStats();
    }

    updateReportsTable() {
        const tableBody = document.getElementById('reports-table-body');
        if (!tableBody) return;

        if (this.filteredData.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center">
                        <i class="fas fa-search"></i>
                        <p>No sightings match the current filters</p>
                    </td>
                </tr>
            `;
            return;
        }

        const tableHTML = this.filteredData
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .map(sighting => this.createReportRow(sighting))
            .join('');

        tableBody.innerHTML = tableHTML;
    }

    createReportRow(sighting) {
        const status = sighting.llmAnalysis?.invasive_status === true ? 'Invasive' : 
                      sighting.llmAnalysis?.invasive_status === false ? 'Native' : 'Unknown';
        const riskLevel = sighting.llmAnalysis?.risk_level || 'Unknown';
        const managementStatus = sighting.management?.status || 'pending';
        
        return `
            <tr>
                <td>${this.formatDate(sighting.timestamp)}</td>
                <td>${sanitizeHtml(sighting.species)}</td>
                <td>
                    <span class="status-badge ${status.toLowerCase()}">${status}</span>
                </td>
                <td>
                    <span class="risk-badge ${riskLevel.toLowerCase()}">${riskLevel}</span>
                </td>
                <td>${this.formatLocation(sighting.latitude, sighting.longitude)}</td>
                <td>${(sighting.confidence * 100).toFixed(1)}%</td>
                <td>
                    <span class="management-badge ${managementStatus}">${managementStatus}</span>
                </td>
                <td>
                    <button class="btn btn-sm btn-primary" onclick="reportsManager.viewDetails('${sighting._id}')">
                        <i class="fas fa-eye"></i> View
                    </button>
                </td>
            </tr>
        `;
    }

    updateReportStats() {
        const totalSightings = this.filteredData.length;
        const invasiveCount = this.filteredData.filter(s => s.llmAnalysis?.invasive_status === true).length;
        const highRiskCount = this.filteredData.filter(s => 
            s.llmAnalysis?.risk_level === 'High' || s.llmAnalysis?.risk_level === 'Critical'
        ).length;
        const pendingCount = this.filteredData.filter(s => s.management?.status === 'pending').length;

        // Update stats display
        const totalElement = document.getElementById('total-sightings-count');
        const invasiveElement = document.getElementById('invasive-sightings-count');
        const highRiskElement = document.getElementById('high-risk-count');
        const pendingElement = document.getElementById('pending-count');

        if (totalElement) totalElement.textContent = totalSightings;
        if (invasiveElement) invasiveElement.textContent = invasiveCount;
        if (highRiskElement) highRiskElement.textContent = highRiskCount;
        if (pendingElement) pendingElement.textContent = pendingCount;
    }

    exportToCSV() {
        if (this.filteredData.length === 0) {
            alert('No data to export');
            return;
        }

        const headers = [
            'Date', 'Species', 'Status', 'Risk Level', 'Location', 
            'Confidence', 'Management Status', 'Description'
        ];

        const csvContent = [
            headers.join(','),
            ...this.filteredData.map(sighting => [
                this.formatDate(sighting.timestamp),
                `"${sighting.species}"`,
                sighting.llmAnalysis?.invasive_status === true ? 'Invasive' : 'Native',
                sighting.llmAnalysis?.risk_level || 'Unknown',
                `"${this.formatLocation(sighting.latitude, sighting.longitude)}"`,
                `${(sighting.confidence * 100).toFixed(1)}%`,
                sighting.management?.status || 'pending',
                `"${sighting.llmAnalysis?.description || ''}"`
            ].join(','))
        ].join('\n');

        this.downloadFile(csvContent, 'sightings-report.csv', 'text/csv');
    }

    exportToPDF() {
        if (this.filteredData.length === 0) {
            alert('No data to export');
            return;
        }

        // Create a simple HTML report for PDF conversion
        const reportHTML = this.generatePDFReport();
        
        // For now, we'll create a downloadable HTML file that can be converted to PDF
        // In a real implementation, you'd use a library like jsPDF or send to a server
        this.downloadFile(reportHTML, 'sightings-report.html', 'text/html');
    }

    generatePDFReport() {
        const reportDate = new Date().toLocaleDateString();
        const totalSightings = this.filteredData.length;
        const invasiveCount = this.filteredData.filter(s => s.llmAnalysis?.invasive_status === true).length;

        return `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Sightings Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .summary { margin-bottom: 20px; }
                    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Plant Recognition Sightings Report</h1>
                    <p>Generated on: ${reportDate}</p>
                </div>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p><strong>Total Sightings:</strong> ${totalSightings}</p>
                    <p><strong>Invasive Species:</strong> ${invasiveCount}</p>
                    <p><strong>Native Species:</strong> ${totalSightings - invasiveCount}</p>
                </div>
                
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Species</th>
                            <th>Status</th>
                            <th>Risk Level</th>
                            <th>Location</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${this.filteredData.map(sighting => `
                            <tr>
                                <td>${this.formatDate(sighting.timestamp)}</td>
                                <td>${sighting.species}</td>
                                <td>${sighting.llmAnalysis?.invasive_status === true ? 'Invasive' : 'Native'}</td>
                                <td>${sighting.llmAnalysis?.risk_level || 'Unknown'}</td>
                                <td>${this.formatLocation(sighting.latitude, sighting.longitude)}</td>
                                <td>${(sighting.confidence * 100).toFixed(1)}%</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </body>
            </html>
        `;
    }

    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    viewDetails(sightingId) {
        const sighting = this.sightingsData.find(s => s._id === sightingId);
        if (!sighting) {
            alert('Sighting not found');
            return;
        }

        // Create and show modal with sighting details
        this.showSightingModal(sighting);
    }

    showSightingModal(sighting) {
        const modalHTML = `
            <div class="modal fade" id="sightingModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Sighting Details</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Basic Information</h6>
                                    <p><strong>Species:</strong> ${sanitizeHtml(sighting.species)}</p>
                                    <p><strong>Date:</strong> ${this.formatDate(sighting.timestamp)}</p>
                                    <p><strong>Location:</strong> ${this.formatLocation(sighting.latitude, sighting.longitude)}</p>
                                    <p><strong>Confidence:</strong> ${(sighting.confidence * 100).toFixed(1)}%</p>
                                </div>
                                <div class="col-md-6">
                                    <h6>Analysis Results</h6>
                                    <p><strong>Status:</strong> ${sighting.llmAnalysis?.invasive_status === true ? 'Invasive' : 'Native'}</p>
                                    <p><strong>Risk Level:</strong> ${sighting.llmAnalysis?.risk_level || 'Unknown'}</p>
                                    <p><strong>NEMBA Category:</strong> ${sighting.llmAnalysis?.advisory_content?.legal_status?.nemba_category || 'Unknown'}</p>
                                    <p><strong>Management Status:</strong> ${sighting.management?.status || 'pending'}</p>
                                </div>
                            </div>
                            ${sighting.llmAnalysis?.description ? `
                                <div class="mt-3">
                                    <h6>Description</h6>
                                    <p>${sanitizeHtml(sighting.llmAnalysis.description)}</p>
                                </div>
                            ` : ''}
                            ${sighting.imageUrl ? `
                                <div class="mt-3">
                                    <h6>Image</h6>
                                    <img src="${sighting.imageUrl}" alt="Sighting" class="img-fluid" style="max-width: 300px;">
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal if any
        const existingModal = document.getElementById('sightingModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add modal to page
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('sightingModal'));
        modal.show();
    }

    formatDate(dateString) {
        if (!dateString) return 'Unknown';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }

    formatLocation(lat, lng) {
        if (!lat || !lng) return 'Unknown';
        return `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
    }
}

// Create singleton instance
export const reportsManager = new ReportsManager(); 