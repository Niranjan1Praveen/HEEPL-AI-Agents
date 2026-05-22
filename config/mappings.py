# server/config/mappings.py

INDUSTRY_CSV_MAPPING = {
    # Dairy & Food Processing
    "dairy": "dairy-food-processing-datasets/dairy_processing.csv",
    "cip": "dairy-food-processing-datasets/food_cleaning_cip.csv",
    "fruit-veg": "dairy-food-processing-datasets/fruit_vegetable_processing.csv",
    "meat": "dairy-food-processing-datasets/meat_poultry_processing.csv",

    # Distilleries
    "molasses": "distilleries/distillery_molasses_wastewater.csv",
    "grain": "distilleries/distillery_grain_wastewater.csv",
    "wineries": "distilleries/distillery_wineries_wastewater.csv",

    # Oil Refineries
    "desalting": "oil-refineries-petroleum/oil_refinery_desalting_wastewater.csv",
    "cracking": "oil-refineries-petroleum/oil_refinery_cracking_wastewater.csv",
    "polymer": "oil-refineries-petroleum/oil_refinery_polymer_wastewater.csv",

    # Pharmaceutical
    "api-bulk": "pharmaceutical-industry/pharma_api_bulk_drug.csv",
    "formulation": "pharmaceutical-industry/pharma_formulation.csv",
    "biologics": "pharmaceutical-industry/pharma_biologics.csv",
    "rd-labs": "pharmaceutical-industry/pharma_rnd_labs.csv",

    # Pulp & Paper
    "chemical-pulping": "pulp-and-paper-mills/pulp_paper_chemical_pulping.csv",
    "bleaching": "pulp-and-paper-mills/pulp_paper_bleaching.csv",
    "recycled": "pulp-and-paper-mills/pulp_paper_recycled_paper.csv",

    # Tannery
    "beamhouse": "tannery-industry-datasets/beamhouse_operations.csv",
    "chrome": "tannery-industry-datasets/chrome_tanning.csv",
    "vegetable": "tannery-industry-datasets/vegetable_tanning.csv",
    "leather-finishing": "tannery-industry-datasets/dyeing_finishing.csv",

    # Textile
    "cotton": "textile-and-dyeing/cotton_processing.csv",
    "synthetic": "textile-and-dyeing/synthetic_textile_processing.csv",
    "wool": "textile-and-dyeing/wool_processing.csv",
    "denim": "textile-and-dyeing/denim_garment_washing.csv",
    "printing": "textile-and-dyeing/textile_printing.csv",

    # Chemicals
    "acid-alkali": "chemicals/Chemical_Acid_Alkali.csv",
    "chlor-alkali": "chemicals/Chemical_Chlor_Alkali.csv",
    "dye-pigments": "chemicals/Chemical_Dye_Pigments.csv",

    # Electroplating
    "electroplating-ops": "electroplating/Electroplating_Electroplating_Ops.csv",
    "acid-pickling-etching": "electroplating/Electroplating_Acid_Pickling.csv",
    "surface-finishing": "electroplating/Electroplating_Surface_Finishing.csv",

    # Fertilizer
    "ammonia": "fertilizer/Fertilizer_Ammonia_Urea.csv",
    "granulation": "fertilizer/Fertilizer_Granulation_Cleaning.csv",
    "nitrate": "fertilizer/Fertilizer_Nitrate.csv",
    "phosphate": "fertilizer/Fertilizer_Phosphate.csv",

    # FMCG
    "cosmetics": "fmcg/FMCG_Cosmetics.csv",
    "home-care": "fmcg/FMCG_Home_Care.csv",
    "personal-care": "fmcg/FMCG_Personal_Care.csv",

    # Iron & Steel
    "blast-furnace": "iron-and-steel-plant/Iron_Steel_Blast_Furnace.csv",
    "coke-ovens": "iron-and-steel-plant/Iron_Steel_Coke_Oven.csv",
    "gas-scrubbing": "iron-and-steel-plant/Iron_Steel_Cooling_Scrubbing.csv",
    "pickling": "iron-and-steel-plant/Iron_Steel_Pickling.csv",
    "rolling-mill": "iron-and-steel-plant/Iron_Steel_Rolling_Mill.csv",

    # Mining
    "amd": "mining-ore/Mining_Acid_Mine_Drainage.csv",
    "flotation": "mining-ore/Mining_Flotation.csv",
    "ore-washing": "mining-ore/Mining_Ore_Washing.csv",

    # Thermal Power
    "ash-handling": "thermal/Thermal_Power_Ash_Handling.csv",
    "boiler-blowdown": "thermal/Thermal_Power_Boiler_Blowdown.csv",
    "cooling-tower": "thermal/Thermal_Power_Cooling_Tower.csv",

    # Slaughterhouse
    "cleaning-sanitation": "slaughterhouse/slaughterhouse_Cleaning_Sanitation.csv",
    "rendering": "slaughterhouse/slaughterhouse_Rendering.csv",
    "slaughtering": "slaughterhouse/slaughterhouse_Slaughtering.csv",

    # Sugar
    "cane-crushing": "sugar/Sugar_Cane_Crushing.csv",
    "clarification": "sugar/Sugar_Clarification.csv",
    "distillery-int": "sugar/Sugar_Distillery_Integration.csv",

    # Pesticides
    "fungicides": "pesticides/Pesticide_Fungicides.csv",
    "herbicides": "pesticides/Pesticide_Herbicides.csv",
    "insecticides": "pesticides/Pesticide_Insecticides.csv",
}

def get_csv_path(industry_id: str) -> str:
    """Get relative CSV path for an industry"""
    return INDUSTRY_CSV_MAPPING.get(industry_id)

def get_all_industry_ids() -> list:
    """Get all available industry IDs"""
    return list(INDUSTRY_CSV_MAPPING.keys())