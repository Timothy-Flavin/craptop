# Re-export everything from the C extension so that:
#   import multi_agent_coverage
#   multi_agent_coverage.BatchedEnvironment(...)  <-- still works
# while also supporting:
#   from multi_agent_coverage.env_wrapper import BatchedGridEnv, FeatureType
from ._core import *
from ._core import FeatureType, BatchedEnvironment

# Global-communication mode (smaller state, no expected_obs / last_agent_locations)
from ._core_global import BatchedEnvironment as BatchedEnvironmentGlobal
