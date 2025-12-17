import os
import time
import logging
import argparse
from typing import Optional
from pathlib import Path
from google import genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# DEFAULT_PROMPT = """
# I'm a Lead data Scientist. I'm trying to learn about Agents and recent LLM developments.
# # Role
# Act as a Principal AI Research Scientist. Your goal is to conduct a "Deep Research" synthesis on the current and future state of Agentic AI, Inter-process Protocols, and Reasoning and Orchestration/Decisiong power of Large Language Models.

# # Objective
# Produce a very detailed technical report that connects the theoretical advancements in reasoning models with the practical engineering of agent protocols and inference optimization. Avoid generic marketing fluff; focus on architecture, papers, learnings, patterns, benchmarks, and engineering trade-offs.

# # Research Dimensions
# Please investigate and research your response across the following pillars:

# ## 1. The Reasoning Engine (System 2 & Inference-Time Compute)
# * **Deep Dive:** Analyze the shift from "training-time compute" to "inference-time compute" (e.g., OpenAI o1, DeepSeek-R1, Google Gemini thinking process).
# * **Mechanisms:** Explain the underlying mechanisms (e.g., Chain of Thought, Tree of Thoughts, Monte Carlo Tree Search, Process Reward Models) that differentiate these from standard autoregressive models.
# * **Trade-offs:** Discuss latency vs. accuracy trade-offs and when "thinking longer" yields diminishing returns.

# ## 2. Agentic Architectures & Protocols
# * **Standards:** Detail the emerging standards for agent-to-agent and agent-to-tool communication. Specifically, analyze the **Model Context Protocol (MCP)** and its implications for interoperability.
# * **Patterns:** Compare Multi-Agent Orchestration frameworks (e.g., AutoGen, LangGraph, CrewAI) vs. Autonomous Single-Agent loops.
# * **State Management:** How are long-horizon agents handling memory and state (e.g., MemGPT, semantic episodic memory)?

# ## 3. Optimization & Efficient Usage
# * **Techniques:** Provide a technical breakdown of how to run these systems efficiently. Cover at least the below topics, (add more important topics as they appear duirng research):
#     * **KV Cache Optimization:** (e.g., PagedAttention, Sliding Windows).
#     * **Speculative Decoding:** Its role in speeding up reasoning chains.
#     * **Quantization:** The impact of FP8/INT4 on reasoning capabilities—does "thinking" degrade with lower precision?
#     * **Prompt Engineering:** "Meta-Prompting" strategies specifically for reasoning models (e.g., constraining the thinking space).
#     * **MCP, A2A Protocols:** How are these protocols evolving and what are the trade-offs?
#     * **Reasoning Models:** How are the reasoning models created, how are they improved, important papers and research on them.
#     * **Orchestration/Decision:** Important papers and research on Orchestration/Decision powers of LLM.
#     * **GRPO/DPO and other RL techniques:** Important papers, findings and research on RL for LLMs.

# """

DEFAULT_PROMPT = """
I want to setup a new PC rig in India. Find me best offers for:
1. ASRock B650E PG Riptide WiFi or similar motherboard like ASUS TUF Gaming B650-PLUS WIFI
2. Ryzen 5 9600X 
3. 2x 16GB DDR5 RAM
4. BenQ PD2706U Monitor
5. 2X RTX 5070 TI
6. Good Power Supply
"""

class DeepResearchAgent:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "deep-research-pro-preview-12-2025"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided via argument or GOOGLE_API_KEY environment variable.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        
    def start_research(self, prompt: str) -> str:
        """Starts the research process and returns the interaction ID."""
        logger.info(f"Starting research with model: {self.model_name}")
        try:
            interaction = self.client.interactions.create(
                input=prompt,
                agent=self.model_name,
                background=True
            )
            logger.info(f"Research started. Interaction ID: {interaction.id}")
            return interaction.id
        except Exception as e:
            logger.error(f"Failed to start research: {e}")
            raise

    def poll_results(self, interaction_id: str, poll_interval: int = 60) -> str:
        """Polls for the results of the research task."""
        logger.info(f"Polling results for interaction ID: {interaction_id}")
        
        while True:
            try:
                interaction = self.client.interactions.get(interaction_id)
                status = interaction.status
                logger.info(f"Current status: {status}")

                if status == "completed":
                    final_report = interaction.outputs[-1].text
                    logger.info("Research completed successfully.")
                    return final_report
                elif status == "failed":
                    logger.error("Research failed.")
                    raise RuntimeError(f"Interaction {interaction_id} failed.")
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error during polling: {e}")
                # Depending on the error, we might want to break or continue. 
                # For now, re-raising to stop execution on API errors.
                raise

    def save_report(self, report: str, output_path: str):
        """Saves the final report to a markdown file."""
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to: {path.absolute()}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Deep Research Agent Runner")
    parser.add_argument("--details", type=str, default=DEFAULT_PROMPT, help="The research prompt or task description.")
    parser.add_argument("--model", type=str, default="deep-research-pro-preview-12-2025", help="The GenAI model to use.")
    parser.add_argument("--output", type=str, default="deep_research_output.md", help="Path to save the output markdown file.")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds to wait between polling status.")
    
    args = parser.parse_args()

    try:
        agent = DeepResearchAgent(api_key="", model_name=args.model)
        interaction_id = agent.start_research(args.details)
        report = agent.poll_results(interaction_id, poll_interval=args.poll_interval)
        agent.save_report(report, args.output)
        
    except ValueError as ve:
        logger.error(f"Configuration Error: {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()