document.addEventListener("DOMContentLoaded", () => {
    const chatWindow = document.getElementById("chat-window");
    const messageForm = document.getElementById("message-form");
    const userInput = document.getElementById("user-input");

    messageForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const userMessage = userInput.value;

        // Add user message as a bubble
        addMessage(userMessage, "user");

        // Clear input field
        userInput.value = "";

        try {
            // Send message to the backend
            const response = await fetch("/send_message", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userMessage }),
            });

            if (!response.ok) throw new Error("Response error");

            const data = await response.json();
            const dispatcherResponse = data.dispatcher_response;

            // Add dispatcher response as a bubble
            addMessage(dispatcherResponse, "dispatcher");
        } catch (error) {
            console.error("Error:", error);
        }
    });

    function addMessage(text, sender) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", sender);
        messageElement.textContent = text;

        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to the latest message
    }
});
