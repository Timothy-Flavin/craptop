#include<iostream>
#include <string>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

#define MAP_SIZE 32
#define N_AGENTS 4
#define VIEW_RANGE 3
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
};
struct AgentObservation{

};

void reset()
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
void update_locations(GameState& s){
    
}

int main(){

    
    return 0;
}
