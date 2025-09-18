#!/usr/bin/env python3
"""Demo script showing device mesh configuration capabilities."""

import jax
from aether.config import Config, DeviceConfig
from aether.utils.device import setup_mesh


def demo_device_mesh_configuration():
    """Demonstrate the new device mesh configuration capabilities."""
    
    print("=== Device Mesh Configuration Demo ===")
    print(f"Available devices: {len(jax.devices())}")
    print(f"Device types: {[d.device_kind for d in jax.devices()]}")
    print()

    # 1. Default behavior (auto-detection)
    print("1. Default auto-detection:")
    default_config = Config()
    print(f"   Auto-detect mesh: {default_config.device.auto_detect_mesh}")
    print(f"   Mesh shape: {default_config.device.mesh_shape}")
    
    mesh = setup_mesh(device_config=default_config.device)
    print(f"   Resulting mesh shape: {mesh.shape}")
    print()

    # 2. Custom mesh configuration
    print("2. Custom mesh configuration:")
    custom_config = DeviceConfig(mesh_shape=(1, 1), auto_detect_mesh=False)
    print(f"   Custom mesh shape: {custom_config.mesh_shape}")
    
    mesh = setup_mesh(device_config=custom_config)
    print(f"   Resulting mesh shape: {mesh.shape}")
    print()

    # 3. Configuration from file
    print("3. Loading configuration from file:")
    try:
        config = Config.from_file("configs/large_model_custom_mesh.json")
        print(f"   Loaded mesh shape: {config.device.mesh_shape}")
        print(f"   Auto detect: {config.device.auto_detect_mesh}")
        print(f"   Model embed_dim: {config.model.embed_dim}")
        
        # Note: This will fail with single device, but shows configuration loading
        print(f"   (Note: This mesh shape requires {config.device.mesh_shape[0] * config.device.mesh_shape[1]} devices)")
    except Exception as e:
        print(f"   Could not create mesh: {e}")
    print()

    # 4. Priority system demonstration
    print("4. Priority system (explicit > config > auto-detect):")
    device_config = DeviceConfig(mesh_shape=(1, 1))
    
    # Explicit shape takes priority
    mesh = setup_mesh(mesh_shape=(1, 1), device_config=device_config)
    print(f"   With explicit shape (1, 1): {mesh.shape}")
    
    # Config shape used when no explicit shape
    mesh = setup_mesh(device_config=device_config)
    print(f"   With config shape: {mesh.shape}")
    
    # Auto-detection when no config
    mesh = setup_mesh()
    print(f"   With auto-detection: {mesh.shape}")
    print()

    print("✓ Device mesh configuration demo completed!")
    print()
    print("Key benefits for fitting bigger models:")
    print("  • Customize tensor and data parallelism dimensions")
    print("  • Optimize memory distribution across devices")
    print("  • Configure mesh topology for better communication")
    print("  • Support for multi-device training setups")


if __name__ == "__main__":
    demo_device_mesh_configuration()