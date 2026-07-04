#!/usr/bin/env python3
"""Check row 7 of test_20."""
import pandas as pd, json

df = pd.read_parquet("data/test_20/train.parquet")
ei = df.iloc[7]["extra_info"]

oracle = ei.get("oracle_calls", "")
oc = json.loads(oracle) if isinstance(oracle, str) else []
actions = [c.get("action", "?") for c in oc]
tools = [c.get("tool_name", "?") for c in oc if c.get("action") == "tool_call"]
sc = ei.get("success_criteria", "")
sc = json.loads(sc) if isinstance(sc, str) else (sc if isinstance(sc, list) else [])
sc_types = [c.get("type", "?") for c in sc] if isinstance(sc, list) else []

print("domain:", ei.get("domain"))
print("scenario:", ei.get("scenario_type"))
print("perturb:", ei.get("perturbation_level"))
print("query:", ei.get("user_query", ""))
print("hidden_tools:", ei.get("hidden_tools"))
print("required_tools:", ei.get("required_tools"))
print("oracle tools:", tools)
print("oracle actions:", actions)
print("success_criteria types:", sc_types)
print("has_missing_function:", ei.get("has_missing_function"))
print("conversation_rounds:", ei.get("conversation_rounds", 1))