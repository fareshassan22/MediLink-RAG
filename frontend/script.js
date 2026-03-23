const API_URL = '';

const chatMessages = document.getElementById('chatMessages');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const roleSelect = document.getElementById('roleSelect');
const modeSelect = document.getElementById('modeSelect');
const pipelineVisualizer = document.getElementById('pipelineVisualizer');
const particlesContainer = document.getElementById('particles');

const stages = ['preprocess', 'retrieve', 'generate', 'judge'];

let isTyping = false;

function createParticles() {
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 10 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
        particlesContainer.appendChild(particle);
    }
}

function activateStage(stageName, status = 'active') {
    const stage = document.querySelector(`[data-stage="${stageName}"]`);
    if (stage) {
        stage.classList.remove('active', 'complete');
        if (status === 'active') {
            stage.classList.add('active');
        } else if (status === 'complete') {
            stage.classList.add('complete');
        }
    }
    
    const stageIndex = stages.indexOf(stageName);
    const connectors = document.querySelectorAll('.pipeline-connector');
    connectors.forEach((conn, i) => {
        if (i < stageIndex + 1) {
            conn.classList.add('active');
        }
    });
}

function resetPipeline() {
    document.querySelectorAll('.pipeline-stage').forEach(stage => {
        stage.classList.remove('active', 'complete');
    });
    document.querySelectorAll('.pipeline-connector').forEach(conn => {
        conn.classList.remove('active');
    });
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2a10 10 0 1 0 10 10H12V2z"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addMessage(content, type, metadata = {}) {
    hideTypingIndicator();
    
    const welcome = document.querySelector('.welcome-message');
    if (welcome) {
        welcome.style.display = 'none';
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatarSvg = type === 'user' 
        ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10H12V2z"/></svg>';
    
    let messageHtml = `
        <div class="message-avatar">
            ${avatarSvg}
        </div>
        <div class="message-content">
            <div class="message-text">${formatMessage(content)}</div>
    `;
    
    if (metadata.confidence !== undefined || metadata.grounding_score !== undefined) {
        messageHtml += `
            <div class="message-meta">
        `;
        
        if (metadata.confidence !== undefined) {
            messageHtml += `
                <span class="confidence-badge">
                    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                        <polyline points="22 4 12 14.01 9 11.01"/>
                    </svg>
                    ثقة: ${(metadata.confidence * 100).toFixed(0)}%
                </span>
            `;
        }
        
        if (metadata.grounding_score !== undefined) {
            messageHtml += `
                <span class="grounding-badge">
                    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="2" y1="12" x2="22" y2="12"/>
                    </svg>
                    grounding: ${(metadata.grounding_score * 100).toFixed(0)}%
                </span>
            `;
        }
        
        messageHtml += `</div>`;
    }
    
    if (metadata.sources && metadata.sources.length > 0) {
        messageHtml += `
            <div class="sources-list">
                <h4>📚 المصادر:</h4>
                <ul>
                    ${metadata.sources.map(s => `<li>${s}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    messageHtml += `</div>`;
    messageDiv.innerHTML = messageHtml;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function formatMessage(text) {
    return text
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

function addErrorMessage(errorText) {
    hideTypingIndicator();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-text">
                <p style="color: var(--error);">⚠️ ${errorText}</p>
            </div>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

async function sendQuery() {
    if (isTyping) return;
    
    const query = queryInput.value.trim();
    if (!query) return;
    
    isTyping = true;
    sendBtn.disabled = true;
    resetPipeline();
    
    addMessage(query, 'user');
    queryInput.value = '';
    
    showTypingIndicator();
    
    try {
        activateStage('preprocess');
        
        const response = await fetch(`${API_URL}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                role: roleSelect.value,
                mode: modeSelect.value
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        activateStage('preprocess', 'complete');
        activateStage('retrieve', 'active');
        
        setTimeout(() => {
            activateStage('retrieve', 'complete');
            activateStage('generate', 'active');
        }, 500);
        
        setTimeout(() => {
            activateStage('generate', 'complete');
            activateStage('judge', 'active');
        }, 1000);
        
        setTimeout(() => {
            activateStage('judge', 'complete');
            
            addMessage(data.answer, 'assistant', {
                confidence: data.confidence,
                grounding_score: data.grounding_score,
                sources: data.sources
            });
            
            isTyping = false;
            sendBtn.disabled = false;
        }, 1500);
        
    } catch (error) {
        console.error('Error:', error);
        isTyping = false;
        sendBtn.disabled = false;
        resetPipeline();
        addErrorMessage('حدث خطأ في الاتصال بالخادم. تأكد من تشغيل API.');
    }
}

sendBtn.addEventListener('click', sendQuery);

queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
});

document.querySelectorAll('.quick-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        queryInput.value = btn.dataset.query;
        sendQuery();
    });
});

createParticles();

const style = document.createElement('style');
style.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
`;
document.head.appendChild(style);
