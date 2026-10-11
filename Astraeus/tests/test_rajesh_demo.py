# tests/test_rajesh_demo.py — Automated verification of 72h Collision & Avoidance Demo for Rajesh
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from app import app
from avoidance_maneuver import compute_fuel_consumption

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_rajesh_demo_endpoint_returns_valid_structure(client):
    """Test that /api/demo/rajesh-collision returns a valid 72h collision and avoidance scenario."""
    res = client.get('/api/demo/rajesh-collision')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert 'scenario' in data
    
    scenario = data['scenario']
    assert 'satellite_1' in scenario
    assert 'satellite_2' in scenario
    assert 'collision' in scenario
    assert 'maneuvers' in scenario
    assert 'summary' in scenario

def test_rajesh_demo_collision_parameters(client):
    """Verify that initial collision event specifies a 72-hour direct collision course."""
    res = client.get('/api/demo/rajesh-collision')
    data = res.get_json()
    coll = data['scenario']['collision']
    
    assert coll['time_to_impact_hours'] == 72.0
    assert coll['miss_distance_km'] == 0.00
    assert coll['collision_probability'] >= 0.90
    assert coll['relative_velocity_km_s'] > 10.0
    assert 'CRITICAL' in coll['risk_level']

def test_rajesh_demo_satellite_capabilities(client):
    """Verify primary satellite is maneuverable and secondary is non-cooperative."""
    res = client.get('/api/demo/rajesh-collision')
    data = res.get_json()
    sat1 = data['scenario']['satellite_1']
    sat2 = data['scenario']['satellite_2']
    
    assert sat1['maneuverable'] is True
    assert sat1['mass_kg'] == 1200.0
    assert sat1['isp_s'] == 300.0
    assert sat1['fuel_capacity_kg'] > 0.0
    
    assert sat2['maneuverable'] is False
    assert sat2['fuel_capacity_kg'] == 0.0

def test_rajesh_demo_maneuver_propellant_and_clearance(client):
    """Verify that applying the optimal maneuver clears the collision threshold and consumes minimal fuel."""
    res = client.get('/api/demo/rajesh-collision')
    data = res.get_json()
    maneuvers = data['scenario']['maneuvers']
    summary = data['scenario']['summary']
    
    radial = maneuvers['radial_outward']
    assert radial['miss_distance_km'] > 25.0  # Safe beyond threshold
    assert radial['miss_distance_km'] >= 50.0 # Standard demo clearance
    assert radial['delta_v_m_s'] == 0.38
    
    # Verify Tsiolkovsky propellant calculation: m0 * (1 - exp(-dv / (isp * g0)))
    expected_fuel = 1200.0 * (1.0 - np.exp(-0.38 / (300.0 * 9.80665)))
    assert np.isclose(radial['fuel_consumed_kg'], expected_fuel, atol=1e-3)
    assert radial['fuel_consumed_kg'] < 0.20  # Less than 200 grams
    assert summary['status'] == 'COLLISION AVERTED'
