from hyperspatial.api import app,health

def test_health_and_openapi():
    assert health()['status']=='ok';schema=app.openapi();assert '/api/v1/adapt' in schema['paths'];assert '/api/v1/design' in schema['paths'];assert '/api/v1/uploads/{filename}' in schema['paths']
