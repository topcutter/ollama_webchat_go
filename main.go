package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"log"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/gorilla/websocket"
	"github.com/ollama/ollama/api"
)

// Define the weather tool for Ollama
var weatherTool = api.Tool{
	Type: "function",
	Function: api.ToolFunction{
		Name:        "get_weather",
		Description: "Get the current weather forecast for a provided location",
		Parameters: struct {
			Type       string   `json:"type"`
			Defs       any      `json:"$defs,omitempty"`
			Items      any      `json:"items,omitempty"`
			Required   []string `json:"required"`
			Properties map[string]struct {
				Type        api.PropertyType `json:"type"`
				Items       any              `json:"items,omitempty"`
				Description string           `json:"description"`
				Enum        []any            `json:"enum,omitempty"`
			} `json:"properties"`
		}{
			Type:     "object",
			Required: []string{"location"},
			Properties: map[string]struct {
				Type        api.PropertyType `json:"type"`
				Items       any              `json:"items,omitempty"`
				Description string           `json:"description"`
				Enum        []any            `json:"enum,omitempty"`
			}{
				"location": {
					Type:        api.PropertyType{"string"},
					Description: "The name of the city for the weather forecast",
				},
			},
		},
	},
}

// handleToolCall processes tool calls from the model
func handleToolCall(toolCall api.ToolCall) string {
	switch toolCall.Function.Name {
	case "get_weather":
		// Extract location from arguments
		location, ok := toolCall.Function.Arguments["location"].(string)
		if !ok {
			return "Error: location parameter is required"
		}
		return getWeatherTool(location)
	default:
		return fmt.Sprintf("Unknown tool: %s", toolCall.Function.Name)
	}
}

var upgrader = websocket.Upgrader{
	// add proper validation logic before deploying
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow connections from any origin, good for testing, bad for security
	},
}

type Message struct {
	Type    string `json:"type"`
	Content string `json:"content"`
	Time    string `json:"time"`
}

// keeps growing with each ollama call so that ai can keep
// track of the conversation
var chatHistory []api.Message

// requiresCurrentInfo analyzes the prompt to determine if it needs real-time/current information
func requiresCurrentInfo(prompt string) bool {
	promptLower := strings.ToLower(prompt)

	// Keywords that indicate need for current information
	currentInfoKeywords := []string{
		"current weather", "weather today", "weather now", "weather in",
		"today's weather", "what's the weather", "how's the weather",
		"temperature in", "temperature at", "temp in",
		"current news", "latest news", "today's news",
		"current time", "what time is it",
		"current date", "what date is it",
		"stock price", "current stock",
		"live", "now", "currently", "today",
		"real-time", "up-to-date",
	}

	for _, keyword := range currentInfoKeywords {
		if strings.Contains(promptLower, keyword) {
			return true
		}
	}

	return false
}

// callOllama sends a user prompt to Ollama using Chat API and returns the response.
// if ollama model requests tool use this is handled internally by the func
// the func won't return data back to the chat client until ollama has 
// reached a 'done' state.
func (app *application) callOllama(prompt string) (string, error) {
	// Parse the Ollama URL
	ollamaURLParsed, err := url.Parse(app.config.ollamaURL)
	if err != nil {
		return "", fmt.Errorf("failed to parse Ollama URL: %v", err)
	}

	// Create Ollama client
	client := api.NewClient(ollamaURLParsed, http.DefaultClient)

	// Add system message if this is the first message
	if len(chatHistory) == 0 {
		systemMessage := api.Message{
			Role: "system",
			Content: `You are a helpful assistant. When you have access to tools, 
			use them to provide accurate, current information.`,
		}
		chatHistory = append(chatHistory, systemMessage)
	}

	// Add user message to chat history
	userMessage := api.Message{
		Role:    "user",
		Content: prompt,
	}
	chatHistory = append(chatHistory, userMessage)

	// Check if the prompt requires current information
	// this is a sanity check to stop the ai from calling tools
	// unless necessary. each model has different tendencies for 
	// how often it tries to call tools
	needsTools := requiresCurrentInfo(prompt)

	app.logger.Debug("Prompt analysis", "need tools", needsTools)

	// Create context
	ctx := context.Background()

	// Create chat request - include tools if needed
	var tools api.Tools
	if needsTools {
		tools = api.Tools{weatherTool}
		app.logger.Debug("Including weather tool in request")
	} else {
		app.logger.Debug("No tools included - using internal knowledge")
	}

	req := &api.ChatRequest{
		Model:    app.config.ollamaModel,
		Messages: chatHistory,
		Stream:   new(bool),
		Tools:    tools,
	}

	// Call Ollama chat API
	var response strings.Builder
	var lastMessage api.Message

	err = client.Chat(ctx, req, func(resp api.ChatResponse) error {
		response.WriteString(resp.Message.Content)
		app.logger.Debug("Ollama", "response", resp.Message.Content)
		lastMessage = resp.Message
		return nil
	})

	if err != nil {
		return "", fmt.Errorf("failed to call Ollama API: %v", err)
	}

	responseContent := strings.TrimSpace(response.String())

	// Handle tool calls if present
	if len(lastMessage.ToolCalls) > 0 {
		app.logger.Debug("Processing tool calls", "tools", len(lastMessage.ToolCalls))

		// Add the assistant's message with tool calls to history
		assistantMessage := api.Message{
			Role:      "assistant",
			Content:   responseContent,
			ToolCalls: lastMessage.ToolCalls,
		}
		chatHistory = append(chatHistory, assistantMessage)

		// Process each tool call
		for _, toolCall := range lastMessage.ToolCalls {
			fnName := toolCall.Function.Name
			fnArgs := toolCall.Function.Arguments

			app.logger.Debug("Processing tool calls", "tool", fnName, "args", fnArgs)

			toolResult := handleToolCall(toolCall)

			// Add tool result as a tool message
			toolMessage := api.Message{
				Role:     "tool",
				Content:  toolResult,
				ToolName: toolCall.Function.Name,
			}
			chatHistory = append(chatHistory, toolMessage)
		}

		// Make another call to get the final response
		finalReq := &api.ChatRequest{
			Model:    app.config.ollamaModel,
			Messages: chatHistory,
			Stream:   new(bool),
			Tools:    api.Tools{weatherTool},
		}

		var finalResponse strings.Builder
		err = client.Chat(ctx, finalReq, func(resp api.ChatResponse) error {
			finalResponse.WriteString(resp.Message.Content)
			app.logger.Debug("ollama", "final response", resp.Message.Content)
			return nil
		})

		if err != nil {
			return "", fmt.Errorf("failed to call Ollama API for final response: %v", err)
		}

		responseContent = strings.TrimSpace(finalResponse.String())
	}

	// Add assistant's final response to chat history
	assistantMessage := api.Message{
		Role:    "assistant",
		Content: responseContent,
	}
	chatHistory = append(chatHistory, assistantMessage)

	return responseContent, nil
}

// chat client page
func (app *application) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		app.logger.Info("Websocket", "upgrade failed", err)
		return
	}
	defer conn.Close()

	app.logger.Info("Web client connected")

	for {
		var msg Message
		err := conn.ReadJSON(&msg)
		if err != nil {
			app.logger.Error(fmt.Sprintf("Error reading message: %v", err))
			break
		}
		app.logger.Debug("Received message", "msg", msg.Content)

		// Call Ollama with the user's message
		ollamaResponse, err := app.callOllama(msg.Content)
		if err != nil {
			app.logger.Error(fmt.Sprintf("Error calling Ollama: %v", err))

			// Send error message to client
			response := Message{
				Type:    "server",
				Content: "Sorry, I'm having trouble connecting to the AI service. Please try again later.",
				Time:    time.Now().Format("15:04:05"),
			}
			conn.WriteJSON(response)
			continue
		}

		// Send back the Ollama response
		response := Message{
			Type:    "server",
			Content: ollamaResponse,
			Time:    time.Now().Format("15:04:05"),
		}

		err = conn.WriteJSON(response)
		if err != nil {
			app.logger.Error(fmt.Sprintf("Error writing message: %v", err))
			break
		}
	}

	app.logger.Info("Client disconnected")
}

// write the home page
func (app *application) handleHome(w http.ResponseWriter, r *http.Request) {
	t := template.Must(template.ParseFiles("index.html"))
	data := struct {
		Time string
	}{
		Time: time.Now().Format("15:04:05"),
	}
	err := t.Execute(w, data)
	if err != nil {
		app.logger.Error(fmt.Sprintf("Template execution error: %v", err))

		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
	}
}

type config struct {
	port        int
	ollamaModel string
	ollamaURL   string
}

type application struct {
	logger *slog.Logger
	config config
}

func main() {
	var cfg config

	// Create a LevelVar to control the log level dynamically
	var levelVar slog.LevelVar
	levelVar.Set(slog.LevelDebug) // set to LevelInfo if you don't want to see 
	// ai response in the terminal

	// Create a logger with a handler that respects the level
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: &levelVar,
	}))

	// command line flags with standard defaults
	flag.IntVar(&cfg.port, "port", 4000, "Web client port")
	flag.StringVar(&cfg.ollamaModel, "LLM", "llama3.1:8b", "Ollama model to use")
	flag.StringVar(&cfg.ollamaURL, "Ollama Server", "http://localhost:11434", "Address of the Ollama server")

	flag.Parse()

	// Declare an instance of the application struct that will
	// be used for dependency injection
	app := &application{
		logger: logger,
		config: cfg,
	}

	http.HandleFunc("/", app.handleHome)
	http.HandleFunc("/ws", app.handleWebSocket)

	httpport := fmt.Sprintf(":%d", app.config.port)
	logger.Info("Starting web server", "Addr", "http://localhost", "Port", httpport)
	logger.Info("Make sure Ollama is running", "Addr", app.config.ollamaURL)
	logger.Info("Current model", "Model", app.config.ollamaModel)

	log.Fatal(http.ListenAndServe(httpport, nil))
}


// provides mock weather data for the location provided by the prompt
// most LLMs expect tools to return JSON. If the information is not
// believeable and relevant to the prompt the tool call will likely fail
func getWeatherTool(location string) string {
	forecast := map[string]any{
		"location": location,
		"forecast": "cloudy",
		"high":     53,
		"unit":     "Fahrenheit",
	}

	forecastJSON, err := json.Marshal(forecast)
	if err != nil {
		return fmt.Sprintf("Error generating forecast data: %v", err)
	}

	return string(forecastJSON)
}
