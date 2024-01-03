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

#ifndef OPEN_SPIEL_GAMES_HARVEST_H_
#define OPEN_SPIEL_GAMES_HARVEST_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// Goofspiel, or the Game of Pure Strategy, is a bidding card game where players
// are trying to obtain the most points. In, Goofspiel(N,K), each player has bid
// cards numbered 1..N and a point card deck containing cards numbered 1..N is
// shuffled and set face-down. There are K turns. Each turn, the top point card
// is revealed, and players simultaneously play a bid card; the point card is
// given to the highest bidder or discarded if the bids are equal. For more
// detail, see: https://en.wikipedia.org/wiki/Goofspiel
//
// This implementation of Goofspiel is slightly more general than the standard
// game. First, more than 2 players can play it. Second, the deck can take on
// pre-determined orders rather than randomly determined. Third, there is an
// option to enable the imperfect information variant described in Sec 3.1.4
// of http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf, where only
// the sequences of wins / losses is revealed (not the players' hands). Fourth,
// players can play for only K turns (if not specified, K=N by default).
//
// The returns_type parameter determines how returns (utilities) are defined:
//   - win_loss distributed 1 point divided by number of winners (i.e. players
//     with highest points), and similarly to -1 among losers
//   - point_difference means each player gets utility as number of points
//     collected minus the average over players.
//   - total_points means each player's return is equal to the number of points
//     they collected.
//
// Parameters:
//   "imp_info"      bool     Enable the imperfect info variant (default: false)
//   "egocentric"   bool     Enable the egocentric info variant (default: false)
//   "num_cards"     int      The highest bid card, and point card (default: 13)
//   "num_turns"     int       The number of turns to play (default: -1, play
//                            for the same number of rounds as there are cards)
//   "players"       int      number of players (default: 2)
//   "points_order"  string   "random" (default), "descending", or "ascending"
//   "returns_type"  string   "win_loss" (default), "point_difference", or
//                            "total_points".

namespace open_spiel {
namespace harvest {

inline constexpr int maxGameLength = 1000;

inline constexpr int appleRadius = 2;
inline constexpr int laserSteps = 25;
inline constexpr int viewWidth = 3;
inline constexpr int viewLength = 7;
inline constexpr int appleReward = 1;
inline constexpr int numPlayers = 2;
inline constexpr int defaultSeed = 1;

enum Unit {
  kempty = -1,
  kapple = -2,
  kwall = -3
};

enum MOVEMENT {
  kLEFT,
  kFORWARD,
  kRIGHT,
  kBACKWARD,
  kTURN_C,
  kTURN_CC,
  kLASER,
  kNO_OP
};


class HarvestObserver;

class HarvestGame; 

class HarvestState : public SimMoveState {
 public:
  explicit HarvestState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;

  std::vector<Action> LegalActions(Player player) const;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override; 
  bool IsTerminal() const override; 

  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override; 
  std::unique_ptr<State> Clone() const override; 

  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;
  std::string InformationStateString(Player player) const override; 
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;

  std::vector<std::vector<int>> getAgentObservations(Player player) const;

  std::string Serialize() const override;

  std::vector<std::vector<int>> grid_; 

  int NumApples() const;

 protected:
  void DoApplyActions(const std::vector<Action>& moves) override;   
  void DoApplyAction(Action action_id) override;

 private:
  std::shared_ptr<const HarvestGame> parent_game_;
  friend class HarvestObserver;

  std::pair<int,int> getHorizontalDirections(std::pair<int,int>& direction) const;
  std::pair<int,int> addPairs(std::pair<int,int> pair1, std::pair<int,int> pair2) const;
  std::pair<int,int> subtractPairs(std::pair<int,int> pair1, std::pair<int,int> pair2) const;
  std::pair<int,int> scalarMultiplyPair(int scale, std::pair<int,int>pair);

  std::vector<float> spawnProbs{0, .005, .02, .05};

  void SpawnApples();
  std::vector<int> LaserAction(Player player);

  std::vector<std::pair<int, int>> appleCheck;
  int numApples = 0;

  std::vector<std::string> mapInput_ = {  "WWWWWWWWW",
                                          "WP     PW",
                                          "W  AAA  W",
                                          "W AAAAA W",
                                          "W  AAA  W", 
                                          "WP     PW",
                                          "WWWWWWWWW"};
  std::vector<std::pair<int, int>> agentLocations_;
  std::vector<std::pair<int, int>> agentDirections_;
  std::pair<int,int> noMovementPair = std::make_pair(0, 0);
  std::vector<int> agentLasered_; 

  std::vector<std::pair<int, int>> agentSpawnPoints_;
  std::vector<std::pair<int, int>> appleSpawnPoints_;

  std::map<std::pair<int,int>, std::map<int, std::pair<int, int>>> changeInPosition_;
  std::vector<std::pair<int,int>> allDirections_;
  std::map<std::pair<int,int>, std::pair<int,int>> cTransition_;
  std::map<std::pair<int,int>, std::pair<int,int>> ccTransition_;

  int x_max_ = mapInput_.size(); 
  int y_max_ = mapInput_[0].size();

  int currentIteration_;
  int isChance_;
  int gameOver_;

  std::vector<double> rewards_; 
  std::vector<double> returns_;
};

class HarvestGame : public Game {
 public:
  explicit HarvestGame(const GameParameters& params);

  std::shared_ptr<Observer> MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const override;

  int NumDistinctActions() const override { return 8; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return numPlayers; }
  double MinUtility() const override;
  double MaxUtility() const override;

  std::string GetRNGState() const;
  void SetRNGState(const std::string& rng_state) const;

  std::mt19937* RNG() const { return rng_.get(); }

  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;

  int MaxGameLength() const override { return maxGameLength; }
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> InformationStateTensorShape() const override;


  std::shared_ptr<Observer> default_observer_;
  std::shared_ptr<Observer> info_state_observer_;


 private:
 mutable std::unique_ptr<std::mt19937> rng_;

};

}  // namespace harvest
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GOOFSPIEL_H_
