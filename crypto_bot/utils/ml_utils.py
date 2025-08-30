"""Machine learning utilities for strategy modules."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def init_ml_or_warn() -> bool:
    """Initialize ML capabilities and return True if available.
    
    Returns:
        bool: True if ML libraries are available, False otherwise.
    """
    try:
        # Try to import common ML libraries
        import sklearn
        import numpy as np
        import pandas as pd
        logger.info("ML libraries available: sklearn, numpy, pandas")
        return True
    except ImportError as e:
        logger.warning(f"ML libraries not available: {e}")
        return False


def load_model(model_name: str) -> Optional[object]:
    """Load a machine learning model by name.
    
    This is a placeholder function. In a real implementation,
    you would load actual trained models from disk or a model registry.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        object: The loaded model, or None if loading fails
    """
    try:
        # Placeholder implementation
        # In production, you would load actual models here
        logger.info(f"Loading model: {model_name}")
        
        # Return a dummy model object that has a predict method
        class DummyModel:
            def predict(self, data):
                """Return a dummy prediction.
                
                Ensures that the input data remains unchanged and returns
                appropriate predictions based on the data type.
                """
                import numpy as np
                import pandas as pd
                
                # Ensure we don't modify the input data
                if isinstance(data, pd.DataFrame):
                    # For DataFrames, return predictions for each row
                    if len(data) == 0:
                        return np.array([])
                    return np.random.random(len(data))
                elif hasattr(data, 'shape'):
                    # For numpy arrays or similar
                    if len(data.shape) == 0:
                        return np.random.random(1)
                    return np.random.random(data.shape[0] if len(data.shape) > 0 else 1)
                elif isinstance(data, (list, tuple)):
                    # For lists or tuples
                    return np.random.random(len(data))
                else:
                    # For single values
                    return np.random.random(1)
            
            def predict_proba(self, data):
                """Return dummy probability predictions.
                
                Ensures that the input data remains unchanged and returns
                appropriate probability predictions based on the data type.
                """
                import numpy as np
                import pandas as pd
                
                # Ensure we don't modify the input data
                if isinstance(data, pd.DataFrame):
                    # For DataFrames, return 2-class probabilities for each row
                    if len(data) == 0:
                        return np.array([]).reshape(0, 2)
                    return np.random.random((len(data), 2))
                elif hasattr(data, 'shape'):
                    # For numpy arrays or similar
                    if len(data.shape) == 0:
                        return np.random.random((1, 2))
                    return np.random.random((data.shape[0] if len(data.shape) > 0 else 1, 2))
                elif isinstance(data, (list, tuple)):
                    # For lists or tuples
                    return np.random.random((len(data), 2))
                else:
                    # For single values
                    return np.random.random((1, 2))
        
        return DummyModel()
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None


def is_ml_available() -> bool:
    """Check if ML capabilities are available.
    
    Returns:
        bool: True if ML is available, False otherwise.
    """
    return init_ml_or_warn()
