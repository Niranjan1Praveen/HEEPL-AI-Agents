"""
config/equipment_registry.py
------------------------------
Industry-specific equipment rosters for the Preventive Maintenance pipeline.
Each key is an industry root ID; sub-category IDs are resolved via SUB_TO_ROOT.
"""

# Maps every sub-category ID to its parent industry root key
SUB_TO_ROOT = {
    # Dairy / Food Processing
    "dairy": "dairy", "cip": "dairy", "fruit-veg": "dairy", "meat": "dairy",
    # Distilleries
    "molasses": "distillery", "grain": "distillery", "wineries": "distillery",
    # Oil Refineries / Petroleum
    "desalting": "refinery", "cracking": "refinery", "polymer": "refinery",
    # Pharmaceutical
    "api-bulk": "pharma", "formulation": "pharma", "biologics": "pharma", "rd-labs": "pharma",
    # Pulp & Paper
    "chemical-pulping": "pulp", "bleaching": "pulp", "recycled": "pulp",
    # Tannery
    "beamhouse": "tannery", "chrome": "tannery", "vegetable": "tannery", "leather-finishing": "tannery",
    # Textile / Dyeing
    "cotton": "textile", "synthetic": "textile", "wool": "textile",
    "denim": "textile", "printing": "textile",
    # Chemicals
    "acid-alkali": "chemical", "chlor-alkali": "chemical", "dye-pigments": "chemical",
    # Electroplating
    "electroplating-ops": "electroplating", "acid-pickling-etching": "electroplating",
    "surface-finishing": "electroplating",
    # Fertilizer
    "ammonia": "fertilizer", "granulation": "fertilizer",
    "nitrate": "fertilizer", "phosphate": "fertilizer",
    # FMCG
    "cosmetics": "fmcg", "home-care": "fmcg", "personal-care": "fmcg",
    # Iron & Steel
    "blast-furnace": "steel", "coke-ovens": "steel", "gas-scrubbing": "steel",
    "pickling": "steel", "rolling-mill": "steel",
    # Mining / Ore
    "amd": "mining", "flotation": "mining", "ore-washing": "mining",
    # Thermal Power
    "ash-handling": "thermal", "boiler-blowdown": "thermal", "cooling-tower": "thermal",
    # Slaughterhouse
    "cleaning-sanitation": "slaughter", "rendering": "slaughter", "slaughtering": "slaughter",
    # Sugar
    "cane-crushing": "sugar", "clarification": "sugar", "distillery-int": "sugar",
    # Pesticides
    "fungicides": "pesticide", "herbicides": "pesticide", "insecticides": "pesticide",
}

EQUIPMENT_REGISTRY = {
    "dairy": [
        {"id": "pump-1",   "name": "CIP Transfer Pump",         "type": "Pump",   "location": "CIP Station"},
        {"id": "pump-2",   "name": "Whey Feed Pump",            "type": "Pump",   "location": "Separator Unit"},
        {"id": "pump-3",   "name": "Effluent Feed Pump",        "type": "Pump",   "location": "ETP Inlet"},
        {"id": "pump-4",   "name": "Sludge Recycle Pump",       "type": "Pump",   "location": "DAF Unit"},
        {"id": "blower-1", "name": "Biogas Blower",             "type": "Blower", "location": "Anaerobic Digester"},
        {"id": "blower-2", "name": "Aeration Blower",           "type": "Blower", "location": "Aerobic Basin"},
    ],
    "distillery": [
        {"id": "pump-1",   "name": "Mash Transfer Pump 1",      "type": "Pump",   "location": "Fermentation Feed"},
        {"id": "pump-2",   "name": "Mash Transfer Pump 2",      "type": "Pump",   "location": "Fermentation Feed"},
        {"id": "pump-3",   "name": "Spent Wash Pump",           "type": "Pump",   "location": "ETP Inlet"},
        {"id": "pump-4",   "name": "Slop Recirculation Pump",   "type": "Pump",   "location": "Evaporator Unit"},
        {"id": "pump-5",   "name": "Condensate Transfer Pump",  "type": "Pump",   "location": "Distillation Column"},
        {"id": "blower-1", "name": "Aeration Blower Primary",   "type": "Blower", "location": "Aerobic Basin"},
        {"id": "blower-2", "name": "Aeration Blower Standby",   "type": "Blower", "location": "Aerobic Basin"},
    ],
    "refinery": [
        {"id": "pump-1",   "name": "Desalter Feed Pump",        "type": "Pump",   "location": "Desalting Unit"},
        {"id": "pump-2",   "name": "Slop Oil Pump",             "type": "Pump",   "location": "API Separator"},
        {"id": "pump-3",   "name": "Effluent Transfer Pump",    "type": "Pump",   "location": "ETP Feed"},
        {"id": "pump-4",   "name": "Cooling Water Pump",        "type": "Pump",   "location": "Heat Exchangers"},
        {"id": "pump-5",   "name": "Condensate Return Pump",    "type": "Pump",   "location": "Boiler Feed"},
        {"id": "blower-1", "name": "Sour Gas Blower",           "type": "Blower", "location": "SRU Unit"},
        {"id": "blower-2", "name": "Aeration Blower",           "type": "Blower", "location": "Biotreatment Basin"},
    ],
    "pharma": [
        {"id": "pump-1",   "name": "API Process Pump",          "type": "Pump",   "location": "Reactor Feed"},
        {"id": "pump-2",   "name": "Solvent Recovery Pump",     "type": "Pump",   "location": "Distillation Unit"},
        {"id": "pump-3",   "name": "Effluent Transfer Pump",    "type": "Pump",   "location": "ETP Inlet"},
        {"id": "pump-4",   "name": "RO Feed Pump",              "type": "Pump",   "location": "Membrane Unit"},
        {"id": "blower-1", "name": "Bio-Scrubber Blower",       "type": "Blower", "location": "Odour Control"},
        {"id": "blower-2", "name": "Aeration Blower",           "type": "Blower", "location": "MBBR Basin"},
    ],
    "pulp": [
        {"id": "pump-1",   "name": "Black Liquor Pump",         "type": "Pump",   "location": "Recovery Boiler Feed"},
        {"id": "pump-2",   "name": "White Liquor Pump",         "type": "Pump",   "location": "Digester Feed"},
        {"id": "pump-3",   "name": "Effluent Feed Pump",        "type": "Pump",   "location": "Primary Clarifier"},
        {"id": "pump-4",   "name": "Sludge Dewatering Pump",   "type": "Pump",   "location": "Belt Press"},
        {"id": "blower-1", "name": "Aeration Blower Primary",   "type": "Blower", "location": "Activated Sludge"},
        {"id": "blower-2", "name": "Chlorine Gas Blower",       "type": "Blower", "location": "Bleaching Tower"},
    ],
    "tannery": [
        {"id": "pump-1",   "name": "Chrome Liquor Pump",        "type": "Pump",   "location": "Tanning Drum Feed"},
        {"id": "pump-2",   "name": "Effluent Sump Pump",        "type": "Pump",   "location": "Beamhouse Drain"},
        {"id": "pump-3",   "name": "Lime Dosing Pump",          "type": "Pump",   "location": "Chemical Dosing"},
        {"id": "pump-4",   "name": "Filter Press Feed Pump",    "type": "Pump",   "location": "Sludge Press"},
        {"id": "blower-1", "name": "Aeration Blower",           "type": "Blower", "location": "Aerobic Basin"},
        {"id": "blower-2", "name": "H2S Scrubber Blower",       "type": "Blower", "location": "Odour Scrubber"},
    ],
    "textile": [
        {"id": "pump-1",   "name": "Dye Liquor Pump",           "type": "Pump",   "location": "Dyeing Vat Feed"},
        {"id": "pump-2",   "name": "Effluent Transfer Pump",    "type": "Pump",   "location": "ETP Inlet"},
        {"id": "pump-3",   "name": "Coagulant Dosing Pump",     "type": "Pump",   "location": "Chemical Dosing"},
        {"id": "pump-4",   "name": "Filter Press Feed Pump",    "type": "Pump",   "location": "Sludge Dewatering"},
        {"id": "pump-5",   "name": "Permeate Pump",             "type": "Pump",   "location": "ZLD Membrane Unit"},
        {"id": "blower-1", "name": "Aeration Blower Primary",   "type": "Blower", "location": "Biotreatment Basin"},
        {"id": "blower-2", "name": "Aeration Blower Standby",   "type": "Blower", "location": "Biotreatment Basin"},
    ],
    "chemical": [
        {"id": "pump-1",   "name": "Acid Transfer Pump",        "type": "Pump",   "location": "Acid Storage"},
        {"id": "pump-2",   "name": "Caustic Dosing Pump",       "type": "Pump",   "location": "Neutralisation Tank"},
        {"id": "pump-3",   "name": "Effluent Feed Pump",        "type": "Pump",   "location": "ETP Inlet"},
        {"id": "pump-4",   "name": "Sludge Pump",               "type": "Pump",   "location": "Clarifier Underflow"},
        {"id": "blower-1", "name": "Fume Scrubber Blower",      "type": "Blower", "location": "HCl Scrubber"},
        {"id": "blower-2", "name": "Aeration Blower",           "type": "Blower", "location": "Aerobic Basin"},
    ],
    "electroplating": [
        {"id": "pump-1",   "name": "Rinse Water Pump",          "type": "Pump",   "location": "Rinse Tank Cascade"},
        {"id": "pump-2",   "name": "Plating Solution Pump",     "type": "Pump",   "location": "Plating Bath"},
        {"id": "pump-3",   "name": "Effluent Transfer Pump",    "type": "Pump",   "location": "ETP Inlet"},
        {"id": "pump-4",   "name": "Chrome Reduction Pump",     "type": "Pump",   "location": "Reduction Tank"},
        {"id": "blower-1", "name": "Fume Extraction Blower",    "type": "Blower", "location": "Plating Line Hood"},
        {"id": "blower-2", "name": "Aeration Blower",           "type": "Blower", "location": "Precipitation Tank"},
    ],
    "fertilizer": [
        {"id": "pump-1",   "name": "Ammonia Feed Pump",         "type": "Pump",   "location": "Synthesis Loop"},
        {"id": "pump-2",   "name": "Scrubber Circulation Pump", "type": "Pump",   "location": "Ammonia Scrubber"},
        {"id": "pump-3",   "name": "Effluent Transfer Pump",    "type": "Pump",   "location": "ETP Inlet"},
        {"id": "pump-4",   "name": "Granulation Return Pump",   "type": "Pump",   "location": "Granulator"},
        {"id": "blower-1", "name": "Ammonia Stripper Blower",   "type": "Blower", "location": "Stripping Column"},
        {"id": "blower-2", "name": "Aeration Blower",           "type": "Blower", "location": "Biotreatment Basin"},
    ],
    "fmcg": [
        {"id": "pump-1",   "name": "Process Transfer Pump",     "type": "Pump",   "location": "Mixing Vessel"},
        {"id": "pump-2",   "name": "CIP Pump",                  "type": "Pump",   "location": "CIP Station"},
        {"id": "pump-3",   "name": "Effluent Feed Pump",        "type": "Pump",   "location": "ETP Inlet"},
        {"id": "pump-4",   "name": "DAF Feed Pump",             "type": "Pump",   "location": "DAF Unit"},
        {"id": "blower-1", "name": "Aeration Blower",           "type": "Blower", "location": "Aerobic Basin"},
        {"id": "blower-2", "name": "Odour Scrubber Blower",     "type": "Blower", "location": "Odour Control"},
    ],
    "steel": [
        {"id": "pump-1",   "name": "Quench Water Pump",         "type": "Pump",   "location": "Coke Oven Battery"},
        {"id": "pump-2",   "name": "Blast Furnace Gas Washer Pump", "type": "Pump", "location": "Gas Scrubbing"},
        {"id": "pump-3",   "name": "Mill Scale Pit Pump",       "type": "Pump",   "location": "Rolling Mill"},
        {"id": "pump-4",   "name": "Pickle Liquor Pump",        "type": "Pump",   "location": "Pickling Line"},
        {"id": "pump-5",   "name": "Effluent Transfer Pump",    "type": "Pump",   "location": "ETP Inlet"},
        {"id": "blower-1", "name": "Sinter Plant Blower",       "type": "Blower", "location": "Sinter Strand"},
        {"id": "blower-2", "name": "Aeration Blower",           "type": "Blower", "location": "Biotreatment Basin"},
    ],
    "mining": [
        {"id": "pump-1",   "name": "AMD Sump Pump 1",           "type": "Pump",   "location": "Mine Drainage Sump"},
        {"id": "pump-2",   "name": "AMD Sump Pump 2",           "type": "Pump",   "location": "Mine Drainage Sump"},
        {"id": "pump-3",   "name": "Flotation Feed Pump",       "type": "Pump",   "location": "Flotation Cell"},
        {"id": "pump-4",   "name": "Tailings Transfer Pump",    "type": "Pump",   "location": "Tailings Dam"},
        {"id": "blower-1", "name": "Flotation Air Blower",      "type": "Blower", "location": "Flotation Cell"},
        {"id": "blower-2", "name": "Neutralisation Blower",     "type": "Blower", "location": "Lime Dosing Tank"},
    ],
    "thermal": [
        {"id": "pump-1",   "name": "Boiler Feed Pump",          "type": "Pump",   "location": "Boiler House"},
        {"id": "pump-2",   "name": "Cooling Tower Pump",        "type": "Pump",   "location": "Cooling Tower Basin"},
        {"id": "pump-3",   "name": "Ash Slurry Pump",           "type": "Pump",   "location": "Ash Handling System"},
        {"id": "pump-4",   "name": "Condensate Extraction Pump","type": "Pump",   "location": "Condenser"},
        {"id": "pump-5",   "name": "Effluent Transfer Pump",    "type": "Pump",   "location": "ETP Inlet"},
        {"id": "blower-1", "name": "FD Fan (Forced Draft)",     "type": "Blower", "location": "Furnace"},
        {"id": "blower-2", "name": "ID Fan (Induced Draft)",    "type": "Blower", "location": "Flue Gas Duct"},
    ],
    "slaughter": [
        {"id": "pump-1",   "name": "Blood Transfer Pump",       "type": "Pump",   "location": "Slaughter Floor"},
        {"id": "pump-2",   "name": "Paunch Manure Pump",        "type": "Pump",   "location": "Stomach Content Tank"},
        {"id": "pump-3",   "name": "DAF Feed Pump",             "type": "Pump",   "location": "DAF Unit"},
        {"id": "pump-4",   "name": "Effluent Transfer Pump",    "type": "Pump",   "location": "ETP Inlet"},
        {"id": "blower-1", "name": "Aeration Blower",           "type": "Blower", "location": "Aerobic Basin"},
        {"id": "blower-2", "name": "Rendering Vent Blower",     "type": "Blower", "location": "Rendering Plant"},
    ],
    "sugar": [
        {"id": "pump-1",   "name": "Juice Transfer Pump",       "type": "Pump",   "location": "Mill Extraction"},
        {"id": "pump-2",   "name": "Molasses Feed Pump",        "type": "Pump",   "location": "Crystalliser Feed"},
        {"id": "pump-3",   "name": "Effluent Transfer Pump",    "type": "Pump",   "location": "ETP Inlet"},
        {"id": "pump-4",   "name": "Press Water Pump",          "type": "Pump",   "location": "Filter Press"},
        {"id": "blower-1", "name": "Aeration Blower",           "type": "Blower", "location": "Aerobic Basin"},
        {"id": "blower-2", "name": "Biogas Compressor Blower",  "type": "Blower", "location": "Anaerobic Digester"},
    ],
    "pesticide": [
        {"id": "pump-1",   "name": "Active Ingredient Pump",    "type": "Pump",   "location": "Synthesis Reactor"},
        {"id": "pump-2",   "name": "Solvent Recovery Pump",     "type": "Pump",   "location": "Distillation Unit"},
        {"id": "pump-3",   "name": "Effluent Feed Pump",        "type": "Pump",   "location": "ETP Inlet"},
        {"id": "pump-4",   "name": "Chemical Dosing Pump",      "type": "Pump",   "location": "Fenton Reactor"},
        {"id": "blower-1", "name": "Fume Scrubber Blower",      "type": "Blower", "location": "Chemical Scrubber"},
        {"id": "blower-2", "name": "Aeration Blower",           "type": "Blower", "location": "Biotreatment Basin"},
    ],
}

_GENERIC_ROSTER = [
    {"id": "pump-1",   "name": "Feed Pump 1",             "type": "Pump",   "location": "ETP Inlet"},
    {"id": "pump-2",   "name": "Feed Pump 2",             "type": "Pump",   "location": "ETP Inlet"},
    {"id": "pump-3",   "name": "Transfer Pump",           "type": "Pump",   "location": "Clarifier"},
    {"id": "pump-4",   "name": "Sludge Pump",             "type": "Pump",   "location": "Sludge Handling"},
    {"id": "blower-1", "name": "Aeration Blower Primary", "type": "Blower", "location": "Aerobic Basin"},
    {"id": "blower-2", "name": "Aeration Blower Standby", "type": "Blower", "location": "Aerobic Basin"},
]


def get_equipment_for_industry(industry_id: str) -> list:
    """
    Return the equipment roster for a given industry ID.
    Resolves sub-category IDs (e.g. 'grain') to their parent root key (e.g. 'distillery').
    Falls back to a generic 4-pump + 2-blower list if no match found.
    """
    root = SUB_TO_ROOT.get(industry_id, industry_id)
    return EQUIPMENT_REGISTRY.get(root, _GENERIC_ROSTER)
