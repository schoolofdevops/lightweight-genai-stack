#!/bin/bash
# Helper script for Lightweight GenAI Stack

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

case "$1" in
    start)
        echo "🚀 Starting Lightweight GenAI Stack..."
        docker compose up -d
        print_status "Stack started!"
        echo ""
        echo "📊 Checking services..."
        sleep 3
        docker compose ps
        echo ""
        echo "📝 Model download in progress. Check with: ./run.sh logs"
        echo "🌐 App will be available at: http://localhost:8501"
        ;;
    
    stop)
        echo "🛑 Stopping stack..."
        docker compose down
        print_status "Stack stopped!"
        ;;
    
    restart)
        echo "🔄 Restarting stack..."
        docker compose restart
        print_status "Stack restarted!"
        ;;
    
    logs)
        echo "📜 Showing logs (Ctrl+C to exit)..."
        docker compose logs -f
        ;;
    
    status)
        echo "📊 Stack Status:"
        docker compose ps
        echo ""
        echo "💾 Memory Usage:"
        docker stats --no-stream
        echo ""
        echo "📦 Ollama Models:"
        docker exec ollama ollama list 2>/dev/null || print_warning "Ollama not ready yet"
        ;;
    
    pull-model)
        if [ -z "$2" ]; then
            echo "Usage: ./run.sh pull-model <model-name>"
            echo "Examples:"
            echo "  ./run.sh pull-model tinyllama:1.1b    # Smallest, ~600MB"
            echo "  ./run.sh pull-model phi3:mini         # Default, ~2.3GB"
            echo "  ./run.sh pull-model llama3.2:3b       # Good balance, ~2GB"
            echo "  ./run.sh pull-model qwen2.5:3b        # Multilingual, ~2GB"
            exit 1
        fi
        echo "📥 Pulling model: $2"
        docker exec ollama ollama pull "$2"
        print_status "Model $2 downloaded!"
        ;;
    
    clean)
        echo "🧹 Cleaning up (removing volumes)..."
        docker compose down -v
        print_status "Clean complete!"
        ;;
    
    memory)
        echo "💾 Real-time Memory Usage (Ctrl+C to exit):"
        docker stats
        ;;
    
    test)
        echo "🧪 Testing Ollama API..."
        curl -s http://localhost:11434/api/generate -d '{
            "model": "phi3:mini",
            "prompt": "Say hello in one sentence.",
            "stream": false
        }' | jq -r '.response' 2>/dev/null || print_error "Test failed. Is Ollama ready?"
        ;;
    
    *)
        echo "Lightweight GenAI Stack Helper"
        echo ""
        echo "Usage: ./run.sh <command>"
        echo ""
        echo "Commands:"
        echo "  start        Start the stack"
        echo "  stop         Stop the stack"
        echo "  restart      Restart all services"
        echo "  logs         Show logs (follow mode)"
        echo "  status       Show status and memory usage"
        echo "  pull-model   Pull a new model (e.g., ./run.sh pull-model llama3.2:3b)"
        echo "  test         Test the Ollama API"
        echo "  memory       Real-time memory monitoring"
        echo "  clean        Stop and remove all data"
        ;;
esac
