#include<iostream>
#include <string>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

#define MAP_SIZE 32
#define N_AGENTS 4
#define VIEW_RANGE 3
#define SPEED 0.5f

constexpr short FLAT_MAP_SIZE = (MAP_SIZE*MAP_SIZE);

struct GameState{
    float expected_danger[MAP_SIZE*MAP_SIZE];
    float actual_danger[MAP_SIZE*MAP_SIZE]; // [discovered 1.0 or 0.0, and danger]
    float observed_danger[N_AGENTS*MAP_SIZE*MAP_SIZE]; // [discovered 1.0 or 0.0, and danger]
    float obs[N_AGENTS*MAP_SIZE*MAP_SIZE];
    float agent_locations[N_AGENTS*2]; 
    
    // What each agent thinks the other agents have covered 
    float expected_obs[N_AGENTS*MAP_SIZE*MAP_SIZE];

    // Where each agent thinks others are r: me, c: other
    float last_agent_locations[N_AGENTS*2*N_AGENTS];

    // if this square has ever been observed for reward purposes
    float previously_observed[N_AGENTS*N_AGENTS];
};

// Set all initial obs to zero then place the agents 
//update obs once for starting state
void reset(GameState& s)
{

}
void update_obs(GameState& s){
    for(short i=0; i<N_AGENTS; ++i){
        short x = short(s.agent_locations[i*2]);
        short y = short(s.agent_locations[i*2+1]);
        for(short ly = y-VIEW_RANGE; ly<y+VIEW_RANGE; ++ly){
            if(ly<0 || ly>=MAP_SIZE) continue;
            for(short lx = x-VIEW_RANGE; lx<x+VIEW_RANGE; ++lx){
                if(lx<0 || lx>=MAP_SIZE) continue;
                s.obs[i*FLAT_MAP_SIZE + y*MAP_SIZE + x] = 1.0f;
                s.observed_danger[i*FLAT_MAP_SIZE + y*MAP_SIZE + x] = s.actual_danger[y*MAP_SIZE + x];
            }
        }
    }
}

// each agent moves in dy dx direction
void update_locations(GameState& s, float* dirs){
    // dirs is 2*N_AGENT long;
    // normalize x y to SPEED length vector then move
}

void step(GameState& s, float* dirs){

}

// For a 32x32 float grid (encoded 1D), find
// the gravitational dx, dy where each grid cell float
// encodes mass. The 1/d^2 is replaced by 1/d^pow
// The reference point for this gravity is x and y where
// each cell's x and y index is the cell's location. 
// min distance is 0.1, less than that and no force.
// x = 2.0, y = 2.0 for cell [1*MAP_SIZE+2] would have
// force gx = 0.0, gy = (1 - 2.0)/(1^pow); This function
// returns the sum over cells
void get_gravity(float* map, int pow, float x, float y){

}

int main(){

    float dirs[N_AGENTS*2];
    float* dirs_p = &dirs[0];
    GameState state;
    reset(state);
    for(short i=0; i<100; ++i){
        // fill dirs with random number

        // fill agent 1 dir with gravity away from expected danger
        // plus gravity away from agent 1's observations to make
        // it leave it's starting location but avoid danger

        // actions are dirs
        step(state, dirs_p);
    }
    
    return 0;
}
