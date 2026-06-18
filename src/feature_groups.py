"""Manufacturing feature group definitions and display metadata."""

FEATURE_GROUPS = {
    "Production": ["ProductionVolume", "ProductionCost"],
    "Supplier and delivery": ["SupplierQuality", "DeliveryDelay"],
    "Quality": ["DefectRate", "QualityScore", "DefectStatus"],
    "Maintenance": ["MaintenanceHours", "DowntimePercentage"],
    "Inventory": ["InventoryTurnover", "StockoutRate"],
    "Workforce and safety": ["WorkerProductivity", "SafetyIncidents"],
    "Energy": ["EnergyConsumption", "EnergyEfficiency"],
    "Additive manufacturing": ["AdditiveProcessTime", "AdditiveMaterialCost"],
}

FEATURE_GROUP_DISPLAY_NAMES = {
    "Production": "Production Scale and Cost",
    "Supplier and delivery": "Supplier and Delivery Reliability",
    "Quality": "Quality Outcomes",
    "Maintenance": "Maintenance and Downtime",
    "Inventory": "Inventory Flow",
    "Workforce and safety": "Workforce and Safety",
    "Energy": "Energy Performance",
    "Additive manufacturing": "Additive Manufacturing",
}

FEATURE_GROUP_DESCRIPTIONS = {
    "Production": "Production volume and cost signals that may show pressure on manufacturing capacity.",
    "Supplier and delivery": "Supplier quality and delivery timing indicators that describe upstream reliability.",
    "Quality": "Direct product-quality measures, including defect rate, quality score, and defect status.",
    "Maintenance": "Maintenance effort and downtime signals that may reflect equipment stability.",
    "Inventory": "Inventory turnover and stockout risk indicators that describe material availability.",
    "Workforce and safety": "Labor productivity and safety incident measures that describe operating conditions.",
    "Energy": "Energy consumption and efficiency measures that describe resource use.",
    "Additive manufacturing": "Process time and material cost measures for additive manufacturing operations.",
}

MODEL_FEATURES = [
    "ProductionVolume",
    "ProductionCost",
    "SupplierQuality",
    "DeliveryDelay",
    "QualityScore",
    "MaintenanceHours",
    "DowntimePercentage",
    "InventoryTurnover",
    "StockoutRate",
    "WorkerProductivity",
    "SafetyIncidents",
    "EnergyConsumption",
    "EnergyEfficiency",
    "AdditiveProcessTime",
    "AdditiveMaterialCost",
]

COMPARISON_FEATURES = [
    "DefectRate",
    "QualityScore",
    "DowntimePercentage",
    "SupplierQuality",
    "MaintenanceHours",
    "WorkerProductivity",
    "DeliveryDelay",
    "StockoutRate",
    "ProductionVolume",
]

TARGET_COLUMN = "DefectStatus"

EXPECTED_COLUMNS = [
    "ProductionVolume",
    "ProductionCost",
    "SupplierQuality",
    "DeliveryDelay",
    "DefectRate",
    "QualityScore",
    "MaintenanceHours",
    "DowntimePercentage",
    "InventoryTurnover",
    "StockoutRate",
    "WorkerProductivity",
    "SafetyIncidents",
    "EnergyConsumption",
    "EnergyEfficiency",
    "AdditiveProcessTime",
    "AdditiveMaterialCost",
    "DefectStatus",
]

NUMERIC_ANALYSIS_COLUMNS = EXPECTED_COLUMNS.copy()
