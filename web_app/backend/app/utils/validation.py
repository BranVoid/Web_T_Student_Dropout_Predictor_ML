def validate_input(data, required_fields):
    missing = [field for field in required_fields if field not in data]
    if missing:
        return {"valid": False, "error": f"Campos faltantes: {missing}"}
    # Añade validaciones de tipo (ej: age es int)
    return {"valid": True}