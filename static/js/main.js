document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const historyList = document.getElementById('history-list');
    const newChatBtn = document.getElementById('new-chat-btn');
    
    let currentChatId = null;

    // *** SỬA LỖI TẠI ĐÂY: Lưu trữ mã SVG của icon để tái sử dụng ***
    const BOT_AVATAR_ICON_SVG = `<svg width="24" height="24" viewBox="0 0 24 24"><path fill="#4285F4" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/><path fill="#34A853" d="M12 4c-2.9 0-5.43 1.56-6.83 3.89L7.1 9.82C7.95 7.93 9.84 6.5 12 6.5s4.05 1.43 4.9 3.32l1.93-1.93C17.43 5.56 14.9 4 12 4z"/><path fill="#FBBC05" d="M4.17 7.89C2.82 9.24 2 10.99 2 12.91h10.5c-.17-2.61-2.24-4.68-4.85-4.85L4.17 7.89z"/><path fill="#EA4335" d="M12 17.5c-2.9 0-5.43-1.56-6.83-3.89l-1.93 1.93C4.57 18.44 7.1 20 10 20c2.61 0 4.96-1.32 6.42-3.34l-2.05-2.05C13.53 16.18 12.18 17.5 10 17.5z"/></svg>`;

    // --- Auto-resize textarea ---
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
    });

    // --- Load chat history on page load ---
    const loadHistory = async () => {
        try {
            const response = await fetch('/history');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const history = await response.json();
            historyList.innerHTML = '';
            history.forEach(chat => {
                const li = document.createElement('li');
                li.classList.add('history-item');
                li.innerHTML = `<a href="#" data-id="${chat.id}">${chat.title}</a>`;
                historyList.appendChild(li);
            });
        } catch (error) {
            console.error('Lỗi khi tải lịch sử:', error);
        }
    };

    // --- Load a specific conversation ---
    const loadConversation = async (chatId) => {
        if (!chatId) return;
        try {
            const response = await fetch(`/conversation/${chatId}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const messages = await response.json();
            chatBox.innerHTML = ''; // Clear chatbox
            messages.forEach(msg => appendMessage(msg.sender, msg.text));
            currentChatId = chatId;
            
            document.querySelectorAll('.history-item a').forEach(a => a.classList.remove('active'));
            const activeLink = document.querySelector(`.history-item a[data-id="${chatId}"]`);
            if (activeLink) activeLink.classList.add('active');
        } catch (error) {
            console.error('Lỗi khi tải cuộc trò chuyện:', error);
        }
    };

    // --- Append a message to the chat box ---
    const appendMessage = (sender, text) => {
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) welcomeMessage.remove();

        const messageWrapper = document.createElement('div');
        messageWrapper.classList.add('message-wrapper', sender);

        const avatar = document.createElement('div');
        avatar.classList.add('message-avatar', sender);
        
        // *** SỬA LỖI TẠI ĐÂY: Sử dụng mã SVG đã lưu trữ ***
        if (sender === 'user') {
            avatar.textContent = 'U';
        } else {
            avatar.innerHTML = BOT_AVATAR_ICON_SVG;
        }
        
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        
        const senderName = document.createElement('div');
        senderName.classList.add('message-sender');
        senderName.textContent = sender === 'user' ? 'Bạn' : 'Chatbot';
        
        const messageText = document.createElement('div');
        messageText.classList.add('message-text');

        try {
            if (typeof marked !== 'undefined' && typeof text === 'string') {
                messageText.innerHTML = marked.parse(text);
            } else {
                messageText.textContent = text;
            }
        } catch (e) {
            console.error("Lỗi khi parse Markdown:", e);
            messageText.textContent = text;
        }

        messageContent.appendChild(senderName);
        messageContent.appendChild(messageText);
        
        messageWrapper.appendChild(avatar);
        messageWrapper.appendChild(messageContent);
        
        chatBox.appendChild(messageWrapper);
        chatBox.scrollTop = chatBox.scrollHeight;
    };
    
    // --- Show loading indicator ---
    const showLoading = () => {
        const loadingWrapper = document.createElement('div');
        loadingWrapper.id = 'loading-indicator';
        loadingWrapper.classList.add('message-wrapper', 'bot');
        loadingWrapper.innerHTML = `
            <div class="message-avatar bot">${BOT_AVATAR_ICON_SVG}</div>
            <div class="message-content">
                <div class="message-sender">Chatbot</div>
                <div class="message-text" style="display: flex; align-items: center; gap: 5px;">
                    <span style="display: inline-block; width: 8px; height: 8px; background-color: #ccc; border-radius: 50%; animation: typing-dots 1.4s infinite;"></span>
                    <span style="display: inline-block; width: 8px; height: 8px; background-color: #ccc; border-radius: 50%; animation: typing-dots 1.4s infinite .2s;"></span>
                    <span style="display: inline-block; width: 8px; height: 8px; background-color: #ccc; border-radius: 50%; animation: typing-dots 1.4s infinite .4s;"></span>
                </div>
            </div>
        `;
        chatBox.appendChild(loadingWrapper);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    // --- Handle form submission ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = userInput.value.trim();
        if (!question) return;

        appendMessage('user', question);
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;

        showLoading();

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, chat_id: currentChatId }),
            });

            const loadingIndicator = document.getElementById('loading-indicator');
            if(loadingIndicator) loadingIndicator.remove();

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Lỗi từ server: ${response.status} ${errorText}`);
            }

            const data = await response.json();

            if (data && data.answer) {
                appendMessage('bot', data.answer);
            } else {
                appendMessage('bot', 'Lỗi: Phản hồi từ máy chủ không hợp lệ.');
            }
            
            if (!currentChatId && data.chat_id) {
                currentChatId = data.chat_id;
                await loadHistory();
                setTimeout(() => {
                    const newChatItem = document.querySelector(`.history-item a[data-id="${currentChatId}"]`);
                    if (newChatItem) newChatItem.classList.add('active');
                }, 100);
            }

        } catch (error) {
            console.error('Đã xảy ra lỗi trong quá trình gửi câu hỏi:', error);
            const loadingIndicator = document.getElementById('loading-indicator');
            if (loadingIndicator) loadingIndicator.remove();
            appendMessage('bot', `Đã có lỗi xảy ra: ${error.message}. Vui lòng thử lại.`);
        } finally {
            sendBtn.disabled = false;
        }
    });

    // --- Event listener for history items ---
    historyList.addEventListener('click', (e) => {
        if (e.target.tagName === 'A') {
            e.preventDefault();
            const chatId = e.target.dataset.id;
            if (chatId !== currentChatId) {
                loadConversation(chatId);
            }
        }
    });

    // --- Event listener for new chat button ---
    newChatBtn.addEventListener('click', () => {
        currentChatId = null;
        chatBox.innerHTML = `
            <div class="welcome-message">
                <svg class="gemini-icon-large" width="48" height="48" viewBox="0 0 24 24"><path fill="#4285F4" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/><path fill="#34A853" d="M12 4c-2.9 0-5.43 1.56-6.83 3.89L7.1 9.82C7.95 7.93 9.84 6.5 12 6.5s4.05 1.43 4.9 3.32l1.93-1.93C17.43 5.56 14.9 4 12 4z"/><path fill="#FBBC05" d="M4.17 7.89C2.82 9.24 2 10.99 2 12.91h10.5c-.17-2.61-2.24-4.68-4.85-4.85L4.17 7.89z"/><path fill="#EA4335" d="M12 17.5c-2.9 0-5.43-1.56-6.83-3.89l-1.93 1.93C4.57 18.44 7.1 20 10 20c2.61 0 4.96-1.32 6.42-3.34l-2.05-2.05C13.53 16.18 12.18 17.5 10 17.5z"/></svg>
                <h1>Xin chào!</h1>
                <p>Tôi có thể giúp gì cho bạn hôm nay?</p>
            </div>
        `;
        document.querySelectorAll('.history-item a').forEach(a => a.classList.remove('active'));
    });
    
    // Add keydown listener to submit with Enter
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Initial load
    loadHistory();
});

// Keyframes for typing animation
const style = document.createElement('style');
style.innerHTML = `
@keyframes typing-dots {
    0%, 20% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
    80%, 100% { transform: translateY(0); }
}`;
document.head.appendChild(style);
