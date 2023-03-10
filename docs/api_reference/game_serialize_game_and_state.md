# OpenSpiel core functions: serialize_game_and_state

[Back to Core API reference](../api_reference.md) \
<br>

`serialize_game_and_state(game: pyspiel.Game, state: pyspiel.State)`

Returns a string representation of the state and the game that created it.

Note: pickle can also be used to serialize / deserialize data, and the pickle
uses the same serialization methods.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
state.apply_action(4)
state.apply_action(2)
state.apply_action(1)
state.apply_action(5)

serialized_data = pyspiel.serialize_game_and_state(game, state)
print(serialized_data)

game_copy, state_copy = pyspiel.deserialize_game_and_state(serialized_data)
print(state_copy)

# Output:
# # Automatically generated by OpenSpiel SerializeGameAndState
# [Meta]
# Version: 1
#
# [Game]
# tic_tac_toe()
# [State]
# 4
# 2
# 1
# 5
#
#
# .xo
# .xo
# ...
```
