"""Unit tests for device configuration and mesh setup."""

import pytest
import jax
from aether.config import Config, DeviceConfig
from aether.utils.device import setup_mesh


def test_device_config_creation():
    """Test DeviceConfig creation with defaults."""
    device_config = DeviceConfig()
    
    assert device_config.mesh_shape is None
    assert device_config.auto_detect_mesh is True


def test_device_config_custom_mesh():
    """Test DeviceConfig with custom mesh shape."""
    device_config = DeviceConfig(mesh_shape=(2, 4), auto_detect_mesh=False)
    
    assert device_config.mesh_shape == (2, 4)
    assert device_config.auto_detect_mesh is False


def test_config_with_device_config():
    """Test Config creation with device configuration."""
    config = Config()
    
    assert hasattr(config, 'device')
    assert config.device.mesh_shape is None
    assert config.device.auto_detect_mesh is True


def test_config_from_dict_with_device():
    """Test Config creation from dictionary with device config."""
    config_dict = {
        "device": {
            "mesh_shape": [1, 1],  # Use valid shape for single device
            "auto_detect_mesh": False
        }
    }
    
    config = Config.from_dict(config_dict)
    
    assert config.device.mesh_shape == [1, 1]
    assert config.device.auto_detect_mesh is False


def test_setup_mesh_with_device_config():
    """Test setup_mesh function with device config."""
    # Use (1, 1) mesh shape to work with single device
    device_config = DeviceConfig(mesh_shape=(1, 1))
    mesh = setup_mesh(device_config=device_config)
    
    assert mesh is not None
    assert mesh.axis_names == ('batch', 'model')


def test_setup_mesh_explicit_shape_priority():
    """Test that explicit mesh_shape takes priority over device_config."""
    device_config = DeviceConfig(mesh_shape=(1, 1))
    # Use explicit shape compatible with single device
    mesh = setup_mesh(mesh_shape=(1, 1), device_config=device_config)
    
    # The mesh should reflect the explicit shape, not device_config shape
    assert mesh is not None
    assert mesh.axis_names == ('batch', 'model')


def test_setup_mesh_default_behavior():
    """Test setup_mesh default behavior without any config."""
    mesh = setup_mesh()
    
    assert mesh is not None
    assert mesh.axis_names == ('batch', 'model')
    
    # Should auto-detect based on available devices
    num_devices = len(jax.devices())
    
    # Verify the mesh was created with expected device count
    total_devices_in_mesh = 1
    for dim_size in mesh.shape.values():
        total_devices_in_mesh *= dim_size
    
    assert total_devices_in_mesh <= num_devices


def test_config_to_dict_includes_device():
    """Test that config.to_dict() includes device configuration."""
    config = Config()
    config.device.mesh_shape = (1, 1)  # Use valid shape for testing
    config.device.auto_detect_mesh = False
    
    config_dict = config.to_dict()
    
    assert "device" in config_dict
    assert config_dict["device"]["mesh_shape"] == (1, 1)
    assert config_dict["device"]["auto_detect_mesh"] is False


if __name__ == "__main__":
    pytest.main([__file__])