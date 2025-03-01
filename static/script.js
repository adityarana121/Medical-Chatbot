function sendMessage() {
    let userInput = document.getElementById("user-input").value.trim();
    if (userInput === "") return;

    let chatBox = document.getElementById("chat-box");

    // Append User Message
    let userMessage = document.createElement("div");
    userMessage.className = "user-message";
    userMessage.textContent = userInput;
    chatBox.appendChild(userMessage);

    // Clear Input
    document.getElementById("user-input").value = "";

    // Scroll Down
    chatBox.scrollTop = chatBox.scrollHeight;

    // Simulate bot typing
    setTimeout(() => {
        let botMessage = document.createElement("div");
        botMessage.className = "bot-message";
        botMessage.textContent = "Thinking...";
        chatBox.appendChild(botMessage);
        chatBox.scrollTop = chatBox.scrollHeight;

        // Simulate response delay
        setTimeout(() => {
            botMessage.textContent = getBotResponse(userInput);
            chatBox.scrollTop = chatBox.scrollHeight;
        }, 1000);
    }, 500);
}

// Sample responses (Replace with AI API)
function getBotResponse(input) {
    input = input.toLowerCase();
    if (input.includes("hello")) return "Hi there! 👋";
    if (input.includes("how are you")) return "I'm just a bot, but I'm doing great! 😊";
    return "I'm still learning! Ask me something else.";
}
