from unicodedata import decimal
from mesa.visualization.ModularVisualization import ModularServer
from model import *
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter
import numpy as np


def agent_portrayal(agent):
    #portrayal = {"Shape": "circle", "Filled": "true", "r": 1}
    portrayal = {"Shape":"rect","w":1,"h":1,"Filled":"true"}
    agent_counts = np.zeros((agent.model.grid.width, agent.model.grid.height))
    for cell in agent.model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count
    pos = agent.traj_queue[0]
    
    if agent_counts[pos[0]][pos[1]] == 1: 
        portrayal['Color'] = '#E6CFE6'
        portrayal['Layer'] = 0
        #portrayal["r"] = 0.2
    elif agent_counts[pos[0]][pos[1]] == 2:
        portrayal['Color'] = '#D94DFF'
        portrayal['Layer'] = 1
        #portrayal["r"] = 0.4
    elif agent_counts[pos[0]][pos[1]] == 3:
        portrayal['Color'] = '#7400A1'
        portrayal['Layer'] = 2
        #portrayal["r"] = 0.9
    return portrayal

grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
chart_nce = ChartModule(
    [{"Label": "Normalized Conditional Entropy", "Color": "#9932CC"}], data_collector_name="datacollector"
)

chart_gain = ChartModule(
    [{"Label": "System Effectiveness Score", "Color": "#8B00FF"}], data_collector_name="datacollector"
)

model_params = {
    "scheme_type":UserSettableParameter(
        "choice",
        "Scheme Type",
        value='self-organized',
        choices=["self-organized", "hybrid", "centralized"],
        description="Scheme Type of Grid Simulation"
    ),
    "N": UserSettableParameter(
        "slider",
        "Number of agents",
        100,
        50,
        150,
        1,
        description="number of agents",
    ),
    "observable_demand_rate": UserSettableParameter(
        "slider",
        "Observation Demand Rate (NA for Self-organized Scheme)",
        0.6,
        0.5,
        1,
        0.1,
        description="The fraction of task values that a decision maker could observe",
    ),
    "change_frequency":UserSettableParameter(
        "slider",
        "Grid Value Change Frequency (Every 100 Steps)",
        3,
        2,
        5,
        1,
        description="Frequency of value changing"
    ),
    "centralized_steps": UserSettableParameter(
        "slider",
        "Number of centralized Steps (Hybrid Scheme Only)",
        1,
        1,
        15,
        5,
        description="Number of centralized steps in a hybrid scheme",
    ),
    "width": 10,
    "height": 10,
    "max_agent_per_cell":3,
    "queue_size":12,
    "p_part":1,
    "neighbor_first":True,
    "grid_with_values": np.random.random((10, 10))*5,
}

server = ModularServer(UserModel, [grid,chart_nce,chart_gain], "User Behavior Model", model_params)
server.port = 8521
server.launch()