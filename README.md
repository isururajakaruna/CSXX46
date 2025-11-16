<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ats-sys/ats">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Reinforcement Learning Trading Agents</h3>
  <h4 align="center">Built on ATS Platform</h4>

  <p align="center">
    Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) implementations for automated cryptocurrency trading
    <br />
    <a href="https://ats-doc.gitbook.io/v1"><strong>Explore ATS Docs »</strong></a>
    <br />
    <br />
    <a href="custom_modules/README.md"><strong>Custom Modules Documentation »</strong></a>
    <br />
    <br />
    <a href="https://github.com/ats-sys/ats">Upstream ATS Repository</a>
    ·
    <a href="custom_modules/strategies/DQN/DQN_README.md">DQN Guide</a>
    ·
    <a href="custom_modules/strategies/PPO/PPO_INFERENCE_GUIDE.md">PPO Guide</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
    </li>
    <li>
      <a href="#base-platform">Base Platform: ATS</a>
    </li>
    <li>
      <a href="#rl-customizations">RL Customizations</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation-local">Installation (Local)</a></li>
        <li><a href="#dual-environment-setup">Dual Environment Setup</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contributors">Contributors</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
<a id="about-the-project"></a>
## About The Project

[![Screen Shot][screenshot]](https://github.com/ats-sys/ats/images/screenshot.png)

This project implements **state-of-the-art Reinforcement Learning algorithms** for automated cryptocurrency trading, leveraging **Deep Q-Networks (DQN)** and **Proximal Policy Optimization (PPO)** agents. The RL agents learn optimal trading strategies through interaction with historical market data, utilizing advanced features including multi-timeframe momentum indicators, volume imbalance signals, and risk-adjusted reward functions.

<a id="base-platform"></a>
## Base Platform: ATS

This project is built upon the **[ATS (Automated Trading System)](https://github.com/ats-sys/ats)** open-source codebase, developed by one of our team members (Isuru S. Rajakaruna) as a hobby project. ATS serves as the **foundational platform for trade execution, backtesting, and exchange connectivity**.

**ATS provides:**
- 🔄 **Backtesting Engine** - High-performance historical data simulation
- 🏦 **Exchange Connectivity** - Binance, Hyperliquid, and custom exchanges
- 📊 **Real-time Plotting** - Live visualization of trading metrics
- 🔌 **RESTful API** - Programmatic job management
- 📈 **Technical Indicators** - MA, RSI, OBV, SuperTrend, and more
- 🎯 **Strategy Framework** - Extensible base classes for custom strategies

**Upstream Repository:** [https://github.com/ats-sys/ats](https://github.com/ats-sys/ats)  
**Official Documentation:** [https://ats-doc.gitbook.io/v1](https://ats-doc.gitbook.io/v1)

<a id="rl-customizations"></a>
## RL Customizations

We have customized ATS to implement advanced **Reinforcement Learning trading agents**:

### 🤖 **DQN (Deep Q-Network) Agent**
- **State Space**: 13-dimensional feature vector including:
  - Price, volume, moving averages, RSI
  - Volume imbalance (buy/sell pressure)
  - Multi-timeframe momentum (1, 5, 20-step returns)
  - Trading frequency control
  - Unrealized P&L tracking
- **Action Space**: BUY, SELL, HOLD
- **Architecture**: 2-layer neural network (128→64→3)
- **Training**: Experience replay buffer with Double DQN
- **Features**: Configurable reward shaping, epsilon-greedy exploration, target network updates

### 🎯 **PPO (Proximal Policy Optimization) Agent**
- **State Space**: Same 13-dimensional feature vector as DQN
- **Action Space**: BUY, SELL, HOLD
- **Architecture**: Actor-Critic with separate policy and value networks
- **Training**: Advantage estimation with clipped surrogate objective
- **Features**: On-policy learning with rollout buffer, GAE for variance reduction

### 📊 **Shared Infrastructure**
- Real-time training monitoring dashboard
- Transition collection and export system
- Comprehensive metrics tracking (rewards, losses, equity)
- Model checkpointing and versioning
- Interactive training scripts with live visualization

**Custom Modules Location:** [`custom_modules/`](custom_modules/)  
**Strategies:** [`custom_modules/strategies/DQN/`](custom_modules/strategies/DQN/) | [`custom_modules/strategies/PPO/`](custom_modules/strategies/PPO/)

<a id="documentation"></a>
## 📚 Documentation

### **RL Implementation Guides**
- [**Custom Modules Overview**](custom_modules/README.md) - Architecture and setup
- [**How to Use Guide**](custom_modules/HOW_TO_USE.md) - Complete usage instructions
- [**DQN Strategy Guide**](custom_modules/strategies/DQN/DQN_README.md) - DQN implementation details
- [**DQN Web Inference**](custom_modules/strategies/DQN/WEB_INFERENCE_GUIDE.md) - Running trained DQN models
- [**PPO Inference Guide**](custom_modules/strategies/PPO/PPO_INFERENCE_GUIDE.md) - Running trained PPO models

### **ATS Platform Documentation**
- [**ATS Official Docs**](https://ats-doc.gitbook.io/v1) - Complete platform documentation
- [**Adding Strategies**](https://ats-doc.gitbook.io/v1/customizations/adding-a-strategy) - Strategy development guide
- [**Adding Indicators**](https://ats-doc.gitbook.io/v1/customizations/adding-an-indicator) - Custom indicator creation
- [**Adding Exchanges**](https://ats-doc.gitbook.io/v1/customizations/adding-a-new-exchange) - Exchange integration

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- BASE PLATFORM -->
## ATS: Automated Trading System

ATS is a comprehensive automated trading system designed for traders and developers. With ATS, you can test, deploy, and manage custom trading strategies across various exchanges through an intuitive interface and API support.

Features:
* **Backtesting:**
  * Deploy strategies for real-time trading on supported exchanges, with live data feeds and execution.
* **Live Trading:**
  * Deploy strategies for real-time trading on supported exchanges, with live data feeds and execution.
* **Custom Strategies:**
  * Develop and integrate unique trading strategies tailored to your requirements. ATS provides a framework for implementing and testing new approaches. 
* **Exchange Connectivity:**
  * Connect to any trading exchange by creating custom exchange classes, enabling flexibility and support for various markets.
* **Reporting:**
  * Generate reports for backtesting and live trading environments. ATS offers insights into performance, risk metrics, and other essential analytics.
* **Custom Indicators:**
  * Create indicators that complement your strategy, providing unique signals and insights based on custom data.
* **Custom Report Generation:**
  * Tailor reports to specific needs, extracting insights from trade performance, risk, and market conditions.
* **API for Trading Jobs:**
  * Use ATS APIs to create, manage, and monitor trading jobs programmatically, providing automation capabilities.
Use ATS APIs to create, manage, and monitor trading jobs programmatically, providing automation capabilities. 
* **Simple Trading UI:**
  * A user-friendly interface for managing trades, viewing strategy performance, and monitoring real-time positions.
Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have contributed to expanding this template!

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
<a id="getting-started"></a>
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

<a id="prerequisites"></a>
### Prerequisites

Following are the prerequisites you need to have beforehand.
* conda (Recommended)
  * You should have set up [Miniconda](https://docs.anaconda.com/miniconda/install/) (or [Anaconda](https://docs.anaconda.com/anaconda/install/)) setup on your system.
* [MongoDB](https://www.mongodb.com/docs/manual/administration/install-community/)
  * Makesure the MongoDB server is running on the system

### Installation (Local)

<a id="setup-option-1"></a>
#### Option 1 - conda:
  * simply run the `setup.sh` (for Linux) or `setup.bat` (for Windows) script to create a separate conda environment named `ats` with all the requirements needed for ATS. 
  * This will guid you through the installation with optional packages for AI-based strategies as well.

  * For Linux  
  ```shell
    ./setup.sh
  ```
  * For Windows
  ```shell
    .\setup.bat 
  ```

<a id="setup-option-2"></a>
#### Option 2 - conda:
  * You can create the conda environment named `ats` with the given `environment.yml` file.
  ```shell
    conda env create -f environment.yml
  ```
 * You may need to install the required AI-related packages manually if you are willing to run AI-based strategies.

<a id="setup-option-3"></a>
#### Option 3 - pip/conda:
* Use the `requirements.txt` file provided to install the required `Python` packages in an existing environment. 
* The package versions in `requirements.txt` have been tested for `Python 3.12` only. 
* If you are using a different `Python` version you may need to check for the compatible package versions.
  ```shell
    pip install -r requirements.txt
  ```
* You may need to install the required AI-related packages manually if you are willing to run AI-based strategies.

### Installation (Docker)
This option will be available soon :star_struck: 

<a id="dual-environment-setup"></a>
### Dual Environment Setup

This project supports **two installation approaches** depending on your needs:

#### **Option A: Base ATS Environment** (Using `requirements.txt` in root)
For running the base ATS platform without RL extensions:
```shell
# From project root
pip install -r requirements.txt
```
- ✅ Lightweight setup
- ✅ Core ATS functionality only
- ✅ No PyTorch or RL dependencies
- **Use when:** Running traditional strategies (MA, RSI-based)

#### **Option B: RL Environment** (Using `custom_modules/requirements.txt`)
For full RL capabilities with DQN/PPO agents:
```shell
# From project root
pip install -r custom_modules/requirements.txt
pip install torch torchvision torchaudio
```
- ✅ Complete RL training stack (237 packages)
- ✅ PyTorch for deep learning
- ✅ Real-time monitoring dashboard
- ✅ All DQN/PPO dependencies
- **Use when:** Training or running RL agents

#### **Recommended: Use Custom Modules Setup Script**
The easiest way to set up the RL environment:
```shell
cd custom_modules
./setup.sh
```
This interactive script will:
- Create a dedicated conda environment
- Install all RL dependencies
- Configure PyTorch (CPU or GPU)
- Set up monitoring tools

**Why Two Dependency Sets?**
- **Base ATS** (`requirements.txt`) - Minimal dependencies for production trading
- **RL Extensions** (`custom_modules/requirements.txt`) - Full ML/DL stack for research and training
- This separation keeps the base platform lightweight while supporting advanced RL features

See [`custom_modules/HOW_TO_USE.md`](custom_modules/HOW_TO_USE.md) for complete setup instructions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

### Run the ATS Server

#### Option 1:
* If you followed setup [Option 1](#setup-option-1) or [Option 2](#setup-option-2), you may use the provided `start.sh` (for Linux) or `start.bat` (for Windows)

* For Linux  
  ```shell
    ./start.sh
  ```
* For Windows
  ```shell
    .\start.bat 
  ```

#### Option 2:
* Inside the created `Python` environment, you can simply run the `main.py` file.
  ```shell
      python -m ats.main
  ```

The ATS server will start on **http://localhost:5010**

### Train RL Agents

Once the ATS server is running, you can train DQN or PPO agents:

#### **DQN Training**
```shell
# Navigate to custom scripts
cd custom_modules/custom_scripts

# Run interactive DQN training
./train_interactive_DQN.sh
```

#### **PPO Training**
```shell
# Navigate to custom scripts
cd custom_modules/custom_scripts

# Run interactive PPO training
./train_interactive_PPO.sh
```

The training dashboard will open at **http://localhost:5050** showing:
- Real-time training loss
- Episode rewards
- Epsilon decay (DQN)
- Live trading metrics

**Trained models are saved to:** `custom_modules/strategies/DQN/saved_models/` or `custom_modules/strategies/PPO/saved_models/`

For complete training guides, see:
- [DQN Training Guide](custom_modules/strategies/DQN/DQN_README.md)
- [Interactive Training Guide](custom_modules/HOW_TO_USE.md)

### Run Trained Models (Inference)

After training, deploy your models for trading:

```shell
# Edit inference configuration
cd custom_modules/strategies/DQN  # or PPO
cp inference_config_web.json my_config.json

# Update model_path in my_config.json
# Set training_mode: false
# Set epsilon: 0.0 for greedy policy

# Submit job via API or web UI at http://localhost:5010
```

See [DQN Web Inference Guide](custom_modules/strategies/DQN/WEB_INFERENCE_GUIDE.md) for detailed instructions.

### Add Custom Modules

* The `custom_modules` directory is dedicated to user-defined customizations. 
* It allows users to add their own exchange connectors, strategies, indicators, and fee models. 
* These modules override the core ones in the ats folder if there is a naming conflict.
* Custom modules should be extended from our [`BaseExchange`][base-exchange-url], [`BaseStrategy`][base-strategy-url], [`BaseIndicator`][base-indicator-url] and [`BaseFees`][base-fees-url] classes

**RL Implementations:**
* [`DQN Strategy`](custom_modules/strategies/DQN/) - Deep Q-Network implementation
* [`PPO Strategy`](custom_modules/strategies/PPO/) - Proximal Policy Optimization implementation

**Documentation:**
* [Project Architecture][docs-architecture]
* [Add new Exchange][docs-add-new-exchange]
* [Add new Strategy][docs-add-new-strategy]
* [Add new Indicator][docs-add-new-indicator]
* [Add new fee structure][docs-add-new-fees]

Refer to [Documentation][docs-url] for complete guidelines.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

### **Base ATS Platform**
- [x] Back trading Engine 
- [x] Live Trading Engine 
- [x] Realtime Plotting API
- [x] Realtime Logging API
- [ ] Add More Exchanges
    - [x] Binance Spot
    - [x] Hyperliquid Spot
    - [ ] Binance Futures
    - [ ] More Exchanges
- [ ] Add More Indicators
    - [x] Volatility (std)
    - [x] Moving Average/ Weighted Moving Average
    - [x] RSI
    - [x] OBV
    - [x] Trend
    - [x] Super Trend
    - [x] AI Indicator
    - [ ] Bollinger Bands
    - [ ] More Indicators
- [ ] Optimize Back trading Engine
- [ ] Release the Docker Image
- [ ] Release the Python Package

### **RL Enhancements (This Project)**
- [x] DQN Implementation
    - [x] Basic DQN with experience replay
    - [x] Double DQN architecture
    - [x] 13-dimensional state space with advanced features
    - [x] Configurable reward shaping
    - [x] Real-time training monitoring
- [x] PPO Implementation
    - [x] Actor-Critic architecture
    - [x] GAE for advantage estimation
    - [x] Clipped surrogate objective
    - [x] Rollout buffer
- [x] Shared Infrastructure
    - [x] Transition collection system
    - [x] Real-time dashboard
    - [x] Interactive training scripts
    - [x] Model checkpointing
- [ ] Advanced Features
    - [ ] Rainbow DQN (C51, Noisy Nets, etc.)
    - [ ] Multi-asset trading support
    - [ ] Portfolio optimization
    - [ ] Advanced reward functions (Sharpe ratio, etc.)
    - [ ] Hyperparameter optimization
    - [ ] Model ensemble techniques
- [ ] Deployment
    - [ ] Production deployment guide
    - [ ] Live trading safety mechanisms
    - [ ] Performance benchmarks vs baselines

See the [upstream ATS issues](https://github.com/ats-sys/ats/issues) for base platform features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the Apache-2.0 License. See [`LICENSE`][license-url] for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTORS -->
## Contributors

- Isuru S. Rajakaruna: [isururajakaruna@gmail.com](mailto:isururajakaruna@gmail.com)
- Sunanda Gamage: [rcsunanda@gmail.com](mailto:rcsunanda@gmail.com)
- Kasun Imesha Wickramasinghe: [iamkasunimesha@gmail.com](mailto:iamkasunimesha@gmail.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

- Project Link: [https://github.com/ats-sys/ats](https://github.com/ats-sys/ats)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[docs-url]: https://ats-doc.gitbook.io/v1
[license-url]: https://github.com/ats-sys/ats?tab=Apache-2.0-1-ov-file
[contributors-url]: https://github.com/ats-sys/ats/graphs/contributors
[docs-architecture]: https://ats-doc.gitbook.io/v1/basics/architecture
[base-exchange-url]: https://github.com/ats-sys/ats/blob/main/ats/exchanges/base_exchange.py
[base-strategy-url]: https://github.com/ats-sys/ats/blob/main/ats/strategies/base_strategy.py
[base-indicator-url]: https://github.com/ats-sys/ats/blob/main/ats/indicators/base_indicator.py
[base-fees-url]: https://github.com/ats-sys/ats/blob/main/ats/exchanges/base_fees.py
[docs-add-new-exchange]: https://ats-doc.gitbook.io/v1/customizations/adding-a-new-exchange
[docs-add-new-strategy]: https://ats-doc.gitbook.io/v1/customizations/adding-a-strategy
[docs-add-new-indicator]: https://ats-doc.gitbook.io/v1/customizations/adding-an-indicator
[docs-add-new-fees]: https://ats-doc.gitbook.io/v1/customizations/custom-fee-structures
[screenshot]: images/screenshot.png
