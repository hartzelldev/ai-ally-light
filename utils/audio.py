import winsound
from core.config_manager import manager

def event_beep(sound_type='info'):
    """
    Plays a system sound if the user has enabled sounds in Global Settings.
    sound_type options: 'info', 'error', 'complete'
    """
    # 1. Access the live config from the manager singleton
    # We use .model_dump() or .get() depending on if it's a Pydantic model or dict
    # Assuming 'enable_sounds' is the key in your GlobalConfig schema
    
    # Check the attribute directly from the Pydantic model in the manager
    try:
        enabled = getattr(manager.config, 'enable_sounds', True)
    except Exception:
        # Fallback if the attribute isn't found
        enabled = True

    if not enabled:
        return

    try:
        if sound_type == 'error':
            # Lower pitch, longer duration for errors (440Hz is A4)
            winsound.Beep(440, 500) 
        elif sound_type == 'complete':
            # High-pitched double-beep for success
            winsound.Beep(1000, 150)
            winsound.Beep(1200, 150)
        else:
            # Standard "Attn" beep
            winsound.Beep(800, 150)
    except Exception:
        # Final fallback: system bell character
        print('\a', end='', flush=True)
        