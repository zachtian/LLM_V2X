import gymnasium as gym
import numpy as np
import time
import json

from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# Import necessary classes from langchain
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

# Define the action mapping to align with the environment's action space
ACTION_MAPPING = {
    'idle': 1,
    'faster': 2,
    'slower': 0
}

def initialize_agents():
    """
    Initialize two ChatOllama agents with structured output using a JSON schema.

    Returns:
    - agent1: The first agent controlling Vehicle 1.
    - agent2: The second agent controlling Vehicle 2.
    """
    # Initialize the base language model
    llm1 = ChatOllama(model="llama3.2:3b")
    llm2 = ChatOllama(model="llama3.2:3b")

    # Define the JSON schema for the agent's response
    json_schema = {
        "title": "AgentResponse",
        "description": "Agent's response containing an action and a message.",
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The action to take, must be one of 'idle', 'faster', 'slower'",
                "enum": ["idle", "faster", "slower"],
                "default": "idle",
            },
            "message": {
                "type": "string",
                "description": "The message to send to the other vehicle",
            },
        },
        "required": ["action", "message"],
    }

    # Use with_structured_output to wrap the LLM with the JSON schema
    structured_llm1 = llm1.with_structured_output(json_schema)
    structured_llm2 = llm2.with_structured_output(json_schema)

    return structured_llm1, structured_llm2

def get_action_and_message_from_agent(agent, vehicle_id, message, goal):
    """
    Generate action and message using the specified agent based on the current state information.
    """
    # Prepare the prompt
    prompt = (
        f"You are an autonomous driving assistant controlling Vehicle {vehicle_id}. "
        f"Considering your goal: \"{goal}\", the message from the other vehicle: \"{message}\""
        f"decide on the appropriate action and message to send back."
    )

    # Use the structured LLM to get the response
    max_attempts = 5  # Maximum number of retries
    attempt = 0

    while attempt < max_attempts:
        try:
            # Use the structured LLM to get the response
            agent_response = agent.invoke(prompt)

            # Check if agent_response is empty or None
            if not agent_response:
                raise ValueError("Received empty response from the agent.")

            # The agent_response is a dictionary matching the JSON schema
            action_word = agent_response['action']
            message_out = agent_response['message']

            # Map the action word to its corresponding integer
            action = ACTION_MAPPING[action_word.lower()]

            return {'action': action, 'message': message_out}

        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt}: Error or empty response: {e}. Retrying...")

    # Default to 'idle' action if all attempts fail
    print("All attempts failed. Defaulting to action 'idle' with an empty message.")
    return {'action': ACTION_MAPPING['idle'], 'message': ''}

def run_simulation():
    """
    Run the simulation using the initialized agents and environment.
    """
    # Initialize the environment
    env = DummyVecEnv([lambda: gym.make('v2x-v0', render_mode='rgb_array')])
    env = VecVideoRecorder(env, "videos", record_video_trigger=lambda step: step % 10 == 0)

    # Initialize the agents
    agent1, agent2 = initialize_agents()

    obs = env.reset()
    done = [False]
    message1 = ''
    message2 = ''

    for i in range(1000):
        if done[0]:
            obs = env.reset()
            done = [False]
            message1 = ''
            message2 = ''
            print(f"Simulation reset at step {i}.")

        # Get actions and messages from both agents
        time.sleep(0.3)
        action_message1 = get_action_and_message_from_agent(agent1, 1, message2, 'Avoid Collision')
        action1 = action_message1['action']
        message1 = action_message1['message']

        time.sleep(0.3)
        action_message2 = get_action_and_message_from_agent(agent2, 2, message1, 'Avoid Collision')
        action2 = action_message2['action']
        message2 = action_message2['message']

        # Log actions and messages
        print(f"Step {i}:")
        print(f"Agent 1 Action: {action1}, Message: {message1}")
        print(f"Agent 2 Action: {action2}, Message: {message2}")

        # Prepare actions for the environment
        actions = [(action1, action2)]

        # Step the environment
        try:
            obs, reward, done, info = env.step(actions)
        except Exception as e:
            print(f"Error during environment step: {e}")
            break  # Exit the simulation on error

        # Render the environment
        env.render()

    env.close()

if __name__ == "__main__":
    run_simulation()
