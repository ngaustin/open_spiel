// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/harvest.h"

#include <algorithm>
#include <memory>
#include <ostream>
#include <utility>
#include <cmath>
#include <iostream>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace harvest {
namespace {

const GameType kGameType{
    /*short_name=*/"harvest",
    /*long_name=*/"Harvest",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kSampledStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"players", GameParameter(numPlayers)},
        {"max_game_length", GameParameter(maxGameLength)}, 
        {"apple_radius", GameParameter(appleRadius)},
        {"laser_steps", GameParameter(laserSteps)},
        {"view_width", GameParameter(viewWidth)}, 
        {"view_length", GameParameter(viewLength)},
        {"apple_reward", GameParameter(appleReward)},
        {"rng_seed", GameParameter(defaultSeed)}, 
    },
    /*default_loadable=*/true,
    /*provides_factored_observation_string=*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new HarvestGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

class HarvestObserver : public Observer {
 public:
  explicit HarvestObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type){}

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    const HarvestState& state =
        open_spiel::down_cast<const HarvestState&>(observed_state);
    const HarvestGame& game =
        open_spiel::down_cast<const HarvestGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    if (iig_obs_type_.perfect_recall) {
      std::vector<std::vector<int>> grid = state.grid_; 
      auto out = allocator->Get("state", {(int) grid[0].size() * (int) grid.size() * 4});
      int offset = grid[0].size() * grid.size();
      for (int i = 0; i < grid.size(); i++) {
        for (int j = 0; j < grid[0].size(); j++) {
          int currValue = grid[i][j];
          int currPlace = i * grid[0].size() + j;
          switch(currValue) {
            case kempty:
              out.at(currPlace) = 1;
              break; 
            case kwall:
              out.at(currPlace + offset) = 1;
              break; 
            case kapple:
              out.at(currPlace + 2*offset) = 1; 
              break; 
            default:
              // this SHOULD be an agent 
              SPIEL_CHECK_TRUE(currValue >= 0); 
              // NOTE: This is only valid for 2-player games
              if (currValue == 0) {
                out.at(currPlace + 3*offset) = 1;
              } else {
                out.at(currPlace + 3*offset) = -1;
              }
          }
        }
      }
    } else {
      auto out = allocator->Get("observation", {4 * viewWidth * viewLength});  // TODO: Make this so that it's times 4 for indicators

      if (state.agentLasered_[player]) {
        int totalSize = 4 * viewWidth * viewLength;
        for (int i = 0; i < totalSize; i++) {
          out.at(i) = 1;
        }
      } else {
        std::vector<std::vector<int>> obs = state.getAgentObservations(player);

        int offset = viewWidth * viewLength;

        std::string printOut = "Checking binary observation: ";
        // std::cout << printOut << std::endl;
        for (int i = 0; i < viewLength; i++) {
          for (int j = 0; j < viewWidth; j++) {
            int currPlace = i * viewWidth + j;
            int currValue = obs[i][j];

            switch(currValue) {
              case kempty:
                out.at(currPlace) = 1;
                break; 
              case kwall:
                out.at(currPlace + offset) = 1;
                break; 
              case kapple:
                out.at(currPlace + 2*offset) = 1; 
                break; 
              default:
                // this SHOULD be an agent 
                SPIEL_CHECK_TRUE(currValue >= 0); 
                if (currValue == player) {
                  out.at(currPlace + 3*offset) = 1;
                } else {
                  out.at(currPlace + 3*offset) = -1;
                }
            }
          }
        }
      }

    }


  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    const HarvestState& state =
        open_spiel::down_cast<const HarvestState&>(observed_state);
    const HarvestGame& game =
        open_spiel::down_cast<const HarvestGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());
    std::string result;

    return result;
  }

 private:
  IIGObservationType iig_obs_type_;
};

HarvestState::HarvestState(std::shared_ptr<const Game> game)
    : SimMoveState(game),
      parent_game_(std::static_pointer_cast<const HarvestGame>(game)),
      currentIteration_(1), 
      isChance_(false),
      gameOver_(false) {

      grid_.reserve(x_max_);

      for (int r = 0; r < x_max_; r++){
        grid_.push_back(std::vector<int>(y_max_, kempty));
      }


      // Change the vector size to numPlayers
      rewards_.resize(numPlayers);
      returns_.resize(numPlayers); 
      agentLasered_.resize(numPlayers);
      agentDirections_.resize(numPlayers);

      // Assigns value 0 to all the elements in the vector
      std::fill(rewards_.begin(), rewards_.end(), 0);
      std::fill(returns_.begin(), returns_.end(), 0);
      std::fill(agentLasered_.begin(), agentLasered_.end(), 0);

      char appleIndicator = 'A';
      char playerIndicator = 'P';
      char wallIndicator = 'W';
      for (int i = 0; i < x_max_; i++){
        for (int j = 0; j < y_max_; j++){
          if (mapInput_[i][j] == appleIndicator) {
            grid_[i][j] = kapple; 
            appleSpawnPoints_.push_back(std::pair<int, int>({i, j}));
          }
          if (mapInput_[i][j] == playerIndicator) {
            agentSpawnPoints_.push_back(std::pair<int, int>({i, j}));
          }
          if (mapInput_[i][j] == wallIndicator) {
            grid_[i][j] = kwall;
          }
        }
      }

      // Providing a seed value
	    // srand(seed);

      for (int i=0; i < numPlayers; i++){
        std::mt19937* sampler = parent_game_->RNG();
        float randNumber = ((float) (*sampler)()) / (*sampler).max(); // ((double) rand() / (RAND_MAX));
        std::vector<int> validIndices({});

        for (int j=0; j < agentSpawnPoints_.size(); j++){
          std::pair<int, int> currPair = agentSpawnPoints_[j];
          if (grid_[currPair.first][currPair.second] == kempty) {
            validIndices.push_back(j);
          } 
        }

        int numOptions = validIndices.size();
        int choice = floor(randNumber / (1.0 / numOptions));
        choice = std::max(std::min(choice, numOptions - 1), 0);
        std::pair<int, int> loc = agentSpawnPoints_[validIndices[choice]];

        SPIEL_CHECK_TRUE(grid_[loc.first][loc.second] == kempty);
        grid_[loc.first][loc.second] = i; 
        agentLocations_.push_back(std::pair<int, int>(loc)); // copy 
      }


      std::map<int,std::pair<int, int>> facingDown; 
      std::map<int,std::pair<int, int>> facingUp; 
      std::map<int,std::pair<int, int>> facingRight; 
      std::map<int,std::pair<int, int>> facingLeft;

      const std::pair<int, int> right(0, 1);
      const std::pair<int, int> left(0, -1);
      const std::pair<int, int> up(-1, 0);
      const std::pair<int, int> down(1, 0);

      facingDown = {{kLEFT, right}, {kRIGHT, left}, {kFORWARD, down}, {kBACKWARD, up}}; 
      facingUp = {{kLEFT, left}, {kRIGHT, right}, {kFORWARD, up}, {kBACKWARD, down}}; 
      facingRight = {{kLEFT, up}, {kRIGHT, down}, {kFORWARD, right}, {kBACKWARD, left}}; 
      facingLeft = {{kLEFT, down}, {kRIGHT, up}, {kFORWARD, left}, {kBACKWARD, right}}; 

      changeInPosition_ = {{down, facingDown}, {up, facingUp}, {right, facingRight}, {left, facingLeft}};
      allDirections_ = {right, left, up, down};

      cTransition_ = {{down, left}, {left, up}, {up, right}, {right, down}};
      ccTransition_ = {{down, right}, {right, up}, {up, left}, {left, down}};

      for (int i = 0; i < numPlayers; i++) {
        std::mt19937* sampler = parent_game_->RNG();
        int choice = (*sampler)() % 4;
        agentDirections_[i] = allDirections_[choice];
      }

      for (int j = -appleRadius; j <= appleRadius; j++) {
        for (int k = -appleRadius; k <= appleRadius; k++) {
          if (pow((j * j) + (k * k), .5) <= appleRadius) {
            std::pair<int, int> addPoint(j, k);
            appleCheck.push_back(addPoint);
          }
        }
      }

      std::string printOut = "Num apples check: "; 
      // std::cout << printOut << appleCheck.size() << std::endl;
}

int HarvestState::CurrentPlayer() const { 
  if (gameOver_) {
    return kTerminalPlayerId; 
  } 
  if (isChance_) {
    return kChancePlayerId;
  }
  return kSimultaneousPlayerId;
}

std::vector<Action> HarvestState::LegalActions(Player player) const {
  SPIEL_CHECK_TRUE(player >= 0);
  if (gameOver_) {
    return std::vector<Action>();
  }
  if (agentLasered_[player]) {
    return std::vector<Action>({kNO_OP});
  }
  return std::vector<Action>({kLEFT, kFORWARD, kRIGHT, kBACKWARD, kTURN_C, kTURN_CC, kLASER, kNO_OP});
}

void HarvestState::InformationStateTensor(Player player,
                                       absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const HarvestGame& game = open_spiel::down_cast<const HarvestGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void HarvestState::ObservationTensor(Player player, absl::Span<float> values) const {
  ContiguousAllocator allocator(values); 
  const HarvestGame& game = 
    open_spiel::down_cast<const HarvestGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::string HarvestState::ObservationString(Player player) const {
  SPIEL_CHECK_TRUE(player >= 0 && player < numPlayers);
  std::string dummy = "Dummy";
  return dummy;
}

std::pair<int,int> HarvestState::getHorizontalDirections(std::pair<int,int>& direction) const {
  std::pair<int,int> newDirection;
  if (direction.first == 0) {
    newDirection.first = 1;
    newDirection.second = 0;
  } else {
    newDirection.first = 0;
    newDirection.second = 1;
  }
  return newDirection;
}

std::pair<int,int> HarvestState::addPairs(std::pair<int,int> pair1, std::pair<int,int> pair2) const {
  return std::make_pair(pair1.first + pair2.first, pair1.second + pair2.second);
}

std::pair<int,int> HarvestState::subtractPairs(std::pair<int,int> pair1, std::pair<int,int> pair2) const {
  return std::make_pair(pair1.first - pair2.first, pair1.second - pair2.second);
}

std::pair<int,int> HarvestState::scalarMultiplyPair(int scale, std::pair<int,int> pair) {
  return std::make_pair(pair.first * scale, pair.second * scale);
}

std::vector<std::vector<int>> HarvestState::getAgentObservations(Player player) const {
  std::vector<std::vector<int>> obs; 
  obs.resize(viewLength);

  for (int i = 0; i < viewLength; i++) {
    obs[i] = std::vector<int>(viewWidth, kempty);
  }

  
  std::pair<int,int> agentDirection = agentDirections_[player];

  // get horizontal directions
  std::pair<int,int> horizontalDirection = getHorizontalDirections(agentDirection);

  std::pair<int,int> agentLoc = agentLocations_[player];

  SPIEL_CHECK_TRUE(viewWidth % 2 == 1);
 
  // If direction is (1, 0) or (0, -1) then the horizontal direction is scaled by positive 
  // If direction is (-1, 0) or (0, 1) then the horizontal direction is scaled by negative 
  int horizontalSwitch;
  if ((agentDirection.first == 1) || (agentDirection.second == -1)) {
    horizontalSwitch = 1;
  } else {
    horizontalSwitch = -1;
  }

  // Iterate through every row in obs
  int middleColumn = viewWidth / 2; 
    // Keep track of the middle location. Copy from agent location first
  std::pair<int,int> middleLocation(agentLoc.first, agentLoc.second);
  for (int rowIndex = 0; rowIndex < viewLength; rowIndex++) {
    for (int columnIndex = 0; columnIndex < viewWidth; columnIndex++) {
      int fromAgent = columnIndex  - middleColumn; 
      int yGrid = middleLocation.second + horizontalDirection.second * fromAgent * horizontalSwitch; 
      if (!(yGrid >= 0 && yGrid < y_max_)) { continue; }
      int xGrid = middleLocation.first + horizontalDirection.first * fromAgent * horizontalSwitch;
      if (!(xGrid >= 0 && xGrid < x_max_)) { continue; }
      obs[rowIndex][columnIndex] = grid_[xGrid][yGrid];
      
    }
    middleLocation.first = middleLocation.first + agentDirection.first; 
    middleLocation.second = middleLocation.second + agentDirection.second;
  }
  
  /*
  std::pair<int,int> agentLoc = agentLocations_[player];
  int squareViewDelta = viewLength / 2;
  int xLeft = agentLoc.first - squareViewDelta; 
  int xRight = agentLoc.first + squareViewDelta;

  if (xLeft < 0) {
    xRight += (-xLeft);
    xLeft = 0;
  } else {
    if (xRight > x_max_ - 1) {
      int correction = xRight - (x_max_ - 1);
      xRight -= correction; 
      xLeft -= correction;
    }
  }

  int yBottom = agentLoc.second - squareViewDelta;
  int yTop = agentLoc.second + squareViewDelta;

  if (yBottom < 0) {
    yTop += (-yBottom);
    yBottom = 0;
  } else {
    if (yTop > y_max_ - 1) {
      int correction = yTop - (y_max_ - 1);
      yTop -= correction; 
      yBottom -= correction;
    }
  }

  for (int i = 0; i <= xRight - xLeft; i++) {
    for (int j = 0; j <= yTop - yBottom; j++) {
      obs[i][j] = grid_[i+xLeft][j+yBottom];
    }
  }

  //std::string printOut = "Agent Observation: ";
  //std::cout << printOut << player << std::endl; 
  //printOut = "Location";
  //std::cout << printOut << agentLoc << std::endl;

  //printOut = "Observation: ";
  //std::cout << printOut << std::endl;
  //for (int k = 0; k < viewLength; k++) {
  //  std::vector<int> row = obs[k];
  //  std::cout << row << std::endl;
  // } */


  return obs;
}

std::string HarvestState::InformationStateString(Player player) const {
  SPIEL_CHECK_TRUE(player >= 0 && player < numPlayers);
  std::string dummy = "Dummy";
  return dummy;
}


std::string HarvestState::ActionToString(Player player, Action action_id) const {
  switch(action_id){
    case kLEFT: return "LEFT";
    case kFORWARD: return "FORWARD";
    case kRIGHT: return "RIGHT";
    case kBACKWARD: return "BACKWARD";
    case kTURN_C: return "CLOCKWISE";
    case kTURN_CC: return "COUNTER-CLOCKWISE";
    case kLASER: return "LASER";
    case kNO_OP: return "NO_OP";
    default: SPIEL_CHECK_TRUE(false);
  }
}

std::string HarvestState::ToString() const {
  return "Dummy string here";
}

bool HarvestState::IsTerminal() const {
  return gameOver_;
}

std::vector<double> HarvestState::Rewards() const {
  return rewards_;
}

std::vector<double> HarvestState::Returns() const {
  return returns_;
}

std::unique_ptr<State> HarvestState::Clone() const {
  return std::unique_ptr<State>(new HarvestState(*this));
}

void HarvestState::DoApplyAction(Action action_id) {
  if (IsSimultaneousNode()) {
    ApplyFlatJointAction(action_id);
    return;
  }
  // SPIEL_CHECK_TRUE(IsChanceNode());
  // assert(false);
}

void HarvestState::DoApplyActions(const std::vector<Action>& moves) {
  std::fill(rewards_.begin(), rewards_.end(), 0);

  std::vector<std::pair<int, int>> intendedPositions;
  intendedPositions.resize(numPlayers);

  std::unordered_set<Player> playersAffectedByLasers;
  std::map<std::pair<int, int>, int> countIntendedPositions;

  std::string printOut = "Actions: ";
  // std::cout << printOut << moves << std::endl;
  // std::cout << currentIteration_ << std::endl;

  for (int i = 0; i < moves.size(); i++) {
    int a = moves[i];
    if (agentLasered_[i]) {
      SPIEL_CHECK_TRUE(a == kNO_OP);
      continue;
    }

    std::pair<int, int>* currLoc = &agentLocations_[i];
    std::pair<int, int>* delta;
    if (a <= 3){ // when <= 3, corresponds to attempted movement
      std::pair<int, int> direction = agentDirections_[i];
      delta = &changeInPosition_[direction][a];
        
    } else {
      delta = &noMovementPair;
      if (a == kTURN_C) {
        agentDirections_[i] = cTransition_[agentDirections_[i]];
      } else {
        if (a == kTURN_CC) {
          agentDirections_[i] = ccTransition_[agentDirections_[i]];
        } else {
          if (a == kLASER) {
            std::vector<int> playersAffected = LaserAction(i);
            SPIEL_CHECK_TRUE(playersAffected[i] == 0);
            for (int j = 0; j < numPlayers; j++) { if (j != i && playersAffected[j]) {playersAffectedByLasers.insert(j); }}
          }
        }
      }
    }


    std::pair<int, int> intended_next_pos = std::make_pair((*currLoc).first + (*delta).first, (*currLoc).second + (*delta).second);
    intended_next_pos.first = std::min(std::max(intended_next_pos.first, 0), x_max_);
    intended_next_pos.second = std::min(std::max(intended_next_pos.second, 0), y_max_);
    bool collideWall = grid_[intended_next_pos.first][intended_next_pos.second] == kwall;

    if (collideWall) {
      intended_next_pos = std::make_pair((*currLoc).first, (*currLoc).second); 
    }

    intendedPositions[i] = intended_next_pos;
    countIntendedPositions[intended_next_pos] += 1;
  }

  // Make players space empty 
  for (int i = 0; i < numPlayers; i++) {
    // TODO: Add conditional about lasered here?? 
    if (!agentLasered_[i]) {
      std::pair<int, int>* prevPosition = &agentLocations_[i];
      grid_[(*prevPosition).first][(*prevPosition).second] = kempty;
    }
  }

  // TODO: Make effects of laser here 
  if (playersAffectedByLasers.size() < numPlayers) {
    for (std::unordered_set<Player>::iterator itr = playersAffectedByLasers.begin(); itr != playersAffectedByLasers.end(); itr++) {
      SPIEL_CHECK_TRUE(!agentLasered_[*itr]);
      agentLasered_[*itr] = laserSteps;
      // NOTE: agentLocations_[itr*] is still set to its previous location. It just won't affect the grid
    }
  }

  // Check for conflicts. If none, then move the agent. 
  for (int i = 0; i < numPlayers; i++) {
    if (agentLasered_[i]) {
      continue;
    }
    std::pair<int, int> pos = intendedPositions[i]; 

    SPIEL_CHECK_TRUE(countIntendedPositions[pos] >= 1);
    //bool collideAgentOtherWay = false; 
    //for (int j = 0; j < numPlayers; j++) {
    //  if (i != j) {
    //    if (pos.first == intendedPositions[j].first && pos.second == intendedPositions[j].second) {
    //      collideAgentOtherWay = true;
    //    }
    //  }
    //}
    bool collideAgent = countIntendedPositions[pos] > 1;
    // SPIEL_CHECK_TRUE(collideAgentOtherWay == collideAgent);

    std::pair<int, int> prevPos = agentLocations_[i]; 

    if (!collideAgent) {
      if (grid_[pos.first][pos.second] == kapple) {
        rewards_[i] += appleReward;
      }

      grid_[pos.first][pos.second] = i; 
      agentLocations_[i] = pos;
    } else {
      grid_[prevPos.first][prevPos.second] = i;
      agentLocations_[i] = prevPos;
    }
  }

  // TODO: Respawn the lasered agents if at lasered == 1
  // srand(seed);
  for (int i = 0; i < numPlayers; i++) {
    if (agentLasered_[i] == 1) {
      std::mt19937* sampler = parent_game_->RNG();
      float randNumber = ((float) (*sampler)()) / (*sampler).max(); // ((double) rand() / (RAND_MAX));
      std::vector<int> validIndices({});

      for (int j=0; j < agentSpawnPoints_.size(); j++){
        std::pair<int, int>* currPair = &agentSpawnPoints_[j];
        if (grid_[(*currPair).first][(*currPair).second] == kempty) {
          validIndices.push_back(j);
        } 
      }

      int numOptions = validIndices.size();
      int choice = floor(randNumber / (1.0 / numOptions));
      choice = std::max(std::min(choice, numOptions - 1), 0);
      std::pair<int, int> loc = agentSpawnPoints_[validIndices[choice]];

      SPIEL_CHECK_TRUE(grid_[loc.first][loc.second] == kempty);
      grid_[loc.first][loc.second] = i; 
      agentLocations_[i] = std::pair<int, int>(loc);

      choice = (*sampler)() % 4;
      agentDirections_[i] = allDirections_[choice];
    }
  }

  // TODO: Update the agentLasered_ tracker
  for (int i = 0; i < numPlayers; i++) {
    //printOut = "directions: "; 
    // std::cout << printOut << std::endl;
    // std::cout << agentDirections_[i] << std::endl;
    agentLasered_[i] = std::max(0, agentLasered_[i] - 1);
    if (agentLasered_[i]) {
      SPIEL_CHECK_TRUE(grid_[agentLocations_[i].first][agentLocations_[i].second] != i);
    }
  } 
  //printOut = "Lasered: ";
  // std::cout << printOut << agentLasered_ << std::endl;

  //printOut = "Grid:";
  // std::cout << printOut << std::endl;
  // for (int i = 0; i < x_max_; i++) {
  //   std::cout << grid_[i] << std::endl;
  // }

  // update returns
  for (int i = 0; i < numPlayers; i++) {
    returns_[i] += rewards_[i];
  }

  printOut = "Returns: ";
  // std::cout << printOut << returns_ << std::endl << std::endl;

  //for (int i =0; i < numPlayers; i++) {
  //  std::pair<int,int> loc = agentLocations_[i];
  //  if (!agentLasered_[i]) {
  //    SPIEL_CHECK_TRUE(grid_[loc.first][loc.second] == i);
  //  }
  //}

  // spawn apples 
  SpawnApples();

  currentIteration_++;
  
  // Check if apples are gone
  bool applesAreGone = true; 
  for (int i = 0; i < appleSpawnPoints_.size(); i++){
    applesAreGone = applesAreGone && (grid_[appleSpawnPoints_[i].first][appleSpawnPoints_[i].second] != kapple);
  }

  // If apples are gone, then make game over early 
  gameOver_ = (currentIteration_ > maxGameLength) || applesAreGone;
}

std::vector<int> HarvestState::LaserAction(Player player) {
  std::vector<int> playersAffected(numPlayers, 0); // this is now a numPlayers length vector with 0/1 if lasered

  std::vector<int> columnsStillAffected(viewWidth, 1); 

  std::vector<std::vector<int>> obs = getAgentObservations(player); 

  for (int i = 0; i < viewLength; i++) {
    for (int j = 0; j < viewWidth; j++) {
      if (columnsStillAffected[j]) {
        bool agentDetected = obs[i][j] >= 0 && obs[i][j] != player; 
        bool wallDetected = obs[i][j] == kwall;  
        int blocked = int(agentDetected || wallDetected);

        if (blocked) {
          columnsStillAffected[j] = 0;
        }

        if (agentDetected) {
          playersAffected[obs[i][j]] = 1;
        }
      }
    }
  }

  return playersAffected;
}

void HarvestState::SpawnApples() {
  for (int i = 0; i < appleSpawnPoints_.size(); i++) {
    std::pair<int, int>* spawnPoint = &appleSpawnPoints_[i];
    int row = (*spawnPoint).first, col = (*spawnPoint).second;

    if (grid_[row][col] == kempty) {
      int numApples = 0;

      for (int l = 0; l < appleCheck.size(); l++) {
        std::pair<int, int>* place = &appleCheck[l];
        int j = (*place).first, k = (*place).second;
        j = row + j;
        k = col + k; 
        if ((0 <= j && j < x_max_) && (0 <= k && k < y_max_)) {
          if (grid_[j][k] == kapple) {
            numApples += 1;
          }
        }
      }
      float spawnProb = spawnProbs[std::min(numApples, 3)];
      std::mt19937* sampler = parent_game_->RNG();

      float val = (float) (*sampler)();
      if ((val) / (*sampler).max() < spawnProb) {
        grid_[row][col] = kapple;
      }
    } 
  }
}

std::unique_ptr<State> HarvestGame::NewInitialState() const {
  return std::unique_ptr<HarvestState>(new HarvestState(shared_from_this())); 
  // std::make_unique<HarvestState>(shared_from_this());
}

std::string HarvestState::Serialize() const {
  if (IsChanceNode()) {
    return "chance";
  } else {
    std::string state_str = "";
    return state_str;
  }
}

std::unique_ptr<State> HarvestGame::DeserializeState(
    const std::string& str) const {
  return NewInitialState();
}

std::vector<int> HarvestGame::InformationStateTensorShape() const {
  return {7 * 9 * 4};
}

std::vector<int> HarvestGame::ObservationTensorShape() const {
  return {viewWidth * viewLength * 4};
}


double HarvestGame::MinUtility() const {
  // Add a small constant here due to numeral issues.
  return 0;
}

double HarvestGame::MaxUtility() const {
  return maxGameLength;
}



HarvestGame::HarvestGame(const GameParameters& params)
    : Game(kGameType, params),
      rng_(new std::mt19937(ParameterValue<int>("rng_seed") == -1 ? std::time(0) : ParameterValue<int>("rng_seed"))) {

  const GameParameters obs_params = {};

  default_observer_ = MakeObserver(kDefaultObsType, obs_params);
  info_state_observer_ = MakeObserver(kInfoStateObsType, obs_params);
}

std::shared_ptr<Observer> HarvestGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {

  return std::make_shared<HarvestObserver>(
      iig_obs_type.value_or(kDefaultObsType));
}

std::string HarvestGame::GetRNGState() const {
  std::ostringstream rng_stream;
  rng_stream << *rng_;
  return rng_stream.str();
}

void HarvestGame::SetRNGState(const std::string& rng_state) const {
  if (rng_state.empty()) return;
  std::istringstream rng_stream(rng_state);
  rng_stream >> *rng_;
}

}  // namespace harvest
}  // namespace open_spiel
