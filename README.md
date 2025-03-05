# Football Match Dynamics: Centralities, Motifs, and Embeddings from Passing Networks

Explore the intricate dynamics of football matches using advanced network analysis techniques. This repository analyzes passing networks to uncover key patterns, player centralities, and structural motifs, providing valuable insights into team performance and strategies.

---

## Features

- **Passing Network Analysis**: Build and analyze passing networks for any football match.
- **Player Centrality Metrics**: Identify key players and their influence on the game.
- **Network Motifs**: Detect recurring structural patterns in passing networks.
- **Embeddings**: Generate feature-rich embeddings for teams and players.
- **Customizable Analysis**: Run the analysis for any match using simple commands.

---

## Sample Outputs

Discover the potential of the tool by checking sample outputs and data in the [Sample Outputs Folder](https://drive.google.com/drive/folders/1edAVOpsfSpvwqDL3s0DXjbt198MkhE0X?usp=sharing).

---

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or later installed.

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/r4stin/Football-Match-Dynamics.git
   cd Football-Match-Dynamics
   ```

2. Move your data files to the `data` folder.

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the analysis for any football match by specifying the home and away teams:

```bash
python -m src.main home_team-away_team
```

### Example

Analyze a match between England and Belgium:

```bash
python -m src.main england-belgium
```

---

## Results

After running the analysis, check the `outputs` folder for detailed results:

- **Centrality Metrics**: Key players and their influence.
- **Network Visualizations**: Interactive and static graphs of passing networks.
- **Motif Analysis**: Insights into recurring structural patterns.
- **Clustering**:  Identify patterns and group teams or players based on their passing networks, revealing similarities and unique playstyles.

---

## Project Structure

```
Football-Match-Dynamics/
├── data/               # Input data files
├── outputs/            # Analysis results
├── src/                # Source code
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
```
