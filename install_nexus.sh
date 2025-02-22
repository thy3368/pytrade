#!/bin/bash

echo "Starting Nexus CLI installation..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Homebrew if not installed
if ! command_exists brew; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew already installed"
fi

# Install Rust if not installed
if ! command_exists rustc; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust already installed"
fi

# Install CMake if not installed
if ! command_exists cmake; then
    echo "Installing CMake..."
    brew install cmake
else
    echo "CMake already installed"
fi

# Add riscv32i target
echo "Adding riscv32i target..."
rustup target add riscv32i-unknown-none-elf

# Install Nexus CLI
echo "Installing Nexus CLI..."
curl https://cli.nexus.xyz/ | sh

echo "Installation completed!"
echo "Please run 'source ~/.bashrc' or start a new terminal session to use Nexus CLI"
