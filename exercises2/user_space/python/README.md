
```
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
```

# bug1  
 
```
 export PYTHONPATH="/workspace/fedml-87/FedML/python"
```

```
python3 ./fedml/computing/scheduler/model_scheduler/device_model_deployment.py
/usr/local/lib/python3.8/dist-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 
  warn(f"Failed to load image Python extension: {e}")
Traceback (most recent call last):
  File "./fedml/computing/scheduler/model_scheduler/device_model_deployment.py", line 35, in <module>
    from ..scheduler_core.compute_cache_manager import ComputeCacheManager
ImportError: attempted relative import with no known parent package
```
改成
```
python3 -m fedml.computing.scheduler.model_scheduler.device_model_deploymen
```