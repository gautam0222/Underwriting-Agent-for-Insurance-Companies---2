// ============================================
// Enhanced Insurance Underwriting Co-Pilot
// Frontend Logic (app.js) - v2.0
// ============================================

// ========== DOM LOOKUPS ==========

// Form + main sections
const form = document.getElementById("predictionForm");
const loadingState = document.getElementById("loadingState");
const resultSection = document.getElementById("resultSection");

// Validation alerts (NEW)
const validationAlert = document.getElementById("validationAlert");
const validationErrors = document.getElementById("validationErrors");
const validationWarnings = document.getElementById("validationWarnings");

// Meta
const modelUsedText = document.getElementById("modelUsedText");
const timestampText = document.getElementById("timestampText");

// Enhanced Risk Display (UPDATED)
const compositeScoreText = document.getElementById("compositeScoreText");
const businessAttractivenessText = document.getElementById("businessAttractivenessText");
const underwritingRiskText = document.getElementById("underwritingRiskText");
const mlProbabilityText = document.getElementById("mlProbabilityText");
const riskScoreText = document.getElementById("riskScoreText");
const scoreFill = document.getElementById("scoreFill");
const confidenceText = document.getElementById("confidenceText");
const tierText = document.getElementById("tierText");
const actionText = document.getElementById("actionText");

// Probabilities
const probNotInterested = document.getElementById("probNotInterested");
const probInterested = document.getElementById("probInterested");

// Risk visuals
const riskBadgeContainer = document.getElementById("riskBadge");
const riskTagsContainer = document.getElementById("riskTags");
const redFlagsContainer = document.getElementById("redFlagsContainer"); // NEW
const redFlagsCount = document.getElementById("redFlagsCount"); // NEW
const redFlagsList = document.getElementById("redFlagsList"); // NEW

// Risk breakdown chart (NEW)
const riskBreakdownChart = document.getElementById("riskBreakdownChart");

// AI insights
const aiSummaryText = document.getElementById("aiSummaryText");
const aiRationaleText = document.getElementById("aiRationaleText");
const aiRecommendationText = document.getElementById("aiRecommendationText");

// Documents
const docCount = document.getElementById("docCount");
const docUrgency = document.getElementById("docUrgency");
const docReviewType = document.getElementById("docReviewType");
const documentsList = document.getElementById("documentsList");
const docEta = document.getElementById("docEta");
const docSpecialInstructions = document.getElementById("docSpecialInstructions"); // NEW

// Chat panel
const chatPanel = document.getElementById("chatPanel");
const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");

// Context for chat (latest assessment)
let lastAssessmentContext = null;

// =====================================================
// UTILITIES
// =====================================================

const safe = (v, d = "—") => (v ?? d);

function showLoading() {
    if (!loadingState) return;
    loadingState.classList.add("active");
}

function hideLoading() {
    if (!loadingState) return;
    loadingState.classList.remove("active");
}

function showResults() {
    if (!resultSection) return;
    resultSection.classList.add("active");
}

function safePercent(value) {
    if (value === null || value === undefined || isNaN(value)) return "0%";
    // Handle both 0-1 and 0-100 ranges
    if (value <= 1) value = value * 100;
    return value.toFixed(1) + "%";
}

function safeText(value, fallback = "—") {
    if (value === null || value === undefined || value === "") return fallback;
    return String(value);
}

function getCurrentTimestamp() {
    try {
        return new Date().toLocaleString();
    } catch (e) {
        return "Just now";
    }
}

// =====================================================
// VALIDATION DISPLAY (NEW)
// =====================================================

function displayValidation(validation) {
    if (!validation) return;

    const errors = validation.errors || [];
    const warnings = validation.warnings || [];

    // Show validation alert if there are errors or warnings
    if ((errors.length > 0 || warnings.length > 0) && validationAlert) {
        validationAlert.style.display = 'block';
        
        // Display errors
        if (validationErrors && errors.length > 0) {
            validationErrors.innerHTML = '<strong>❌ Errors:</strong><ul>' +
                errors.map(err => `<li>${err}</li>`).join('') +
                '</ul>';
            validationErrors.style.display = 'block';
        } else if (validationErrors) {
            validationErrors.style.display = 'none';
        }

        // Display warnings
        if (validationWarnings && warnings.length > 0) {
            validationWarnings.innerHTML = '<strong>⚠️ Warnings:</strong><ul>' +
                warnings.map(warn => `<li>${warn}</li>`).join('') +
                '</ul>';
            validationWarnings.style.display = 'block';
        } else if (validationWarnings) {
            validationWarnings.style.display = 'none';
        }
    } else if (validationAlert) {
        validationAlert.style.display = 'none';
    }
}

// =====================================================
// RED FLAGS DISPLAY (NEW)
// =====================================================

function displayRedFlags(validation) {
    if (!validation || !redFlagsContainer) return;

    const redFlags = validation.red_flags || [];
    const redFlagScore = validation.red_flag_score || 0;

    if (redFlags.length === 0) {
        redFlagsContainer.style.display = 'none';
        return;
    }

    redFlagsContainer.style.display = 'block';

    // Update count
    if (redFlagsCount) {
        redFlagsCount.textContent = `${redFlags.length} Red Flag${redFlags.length !== 1 ? 's' : ''} (Score: ${redFlagScore}/100)`;
    }

    // Render list
    if (redFlagsList) {
        redFlagsList.innerHTML = '';
        redFlags.forEach(flag => {
            const li = document.createElement('li');
            li.className = 'red-flag-item';
            
            // Check if it's a critical flag
            if (flag.includes('🚨') || flag.toLowerCase().includes('critical')) {
                li.classList.add('critical');
            }
            
            li.textContent = flag;
            redFlagsList.appendChild(li);
        });
    }
}

// =====================================================
// RISK BREAKDOWN VISUALIZATION (NEW)
// =====================================================

function displayRiskBreakdown(prediction) {
    if (!prediction || !riskBreakdownChart) return;

    const breakdown = prediction.risk_breakdown;
    if (!breakdown) {
        riskBreakdownChart.style.display = 'none';
        return;
    }

    riskBreakdownChart.style.display = 'block';

    // Extract numeric values from strings like "75.5/100"
    const parseScore = (str) => {
        if (typeof str === 'number') return str;
        const match = String(str).match(/(\d+\.?\d*)/);
        return match ? parseFloat(match[1]) : 0;
    };

    const customerInterest = parseScore(breakdown.customer_interest || 0);
    const underwritingConcerns = parseScore(breakdown.underwriting_concerns || 0);
    const overallRisk = parseScore(breakdown.overall_risk || 0);

    riskBreakdownChart.innerHTML = `
        <div class="risk-breakdown-item">
            <div class="risk-breakdown-label">
                <span>Customer Interest</span>
                <span class="risk-breakdown-value">${customerInterest.toFixed(1)}/100</span>
            </div>
            <div class="risk-breakdown-bar">
                <div class="risk-breakdown-fill" style="width: ${customerInterest}%; background: #10b981;"></div>
            </div>
        </div>
        <div class="risk-breakdown-item">
            <div class="risk-breakdown-label">
                <span>Underwriting Concerns</span>
                <span class="risk-breakdown-value">${underwritingConcerns.toFixed(1)}/100</span>
            </div>
            <div class="risk-breakdown-bar">
                <div class="risk-breakdown-fill" style="width: ${underwritingConcerns}%; background: #ef4444;"></div>
            </div>
        </div>
        <div class="risk-breakdown-item">
            <div class="risk-breakdown-label">
                <span><strong>Overall Risk Score</strong></span>
                <span class="risk-breakdown-value"><strong>${overallRisk.toFixed(1)}/100</strong></span>
            </div>
            <div class="risk-breakdown-bar">
                <div class="risk-breakdown-fill" style="width: ${overallRisk}%; background: ${overallRisk < 30 ? '#10b981' : overallRisk < 55 ? '#f59e0b' : '#ef4444'};"></div>
            </div>
        </div>
    `;
}

// ============================================
// Risk badge + tags render helpers
// ============================================

function renderRiskBadge(tier) {
    if (!riskBadgeContainer) return;
    riskBadgeContainer.innerHTML = "";

    if (!tier) return;

    const tierLower = tier.toString().toLowerCase();
    let badgeClass = "risk-badge--medium";
    let label = tier;
    let icon = "⚠️";

    if (tierLower.includes("low")) {
        badgeClass = "risk-badge--low";
        label = "Low Risk";
        icon = "✅";
    } else if (tierLower.includes("high")) {
        badgeClass = "risk-badge--high";
        label = "High Risk";
        icon = "🚨";
    } else {
        badgeClass = "risk-badge--medium";
        label = "Medium Risk";
        icon = "⚠️";
    }

    const badge = document.createElement("div");
    badge.className = `risk-badge ${badgeClass}`;
    badge.innerHTML = `
        <span class="risk-badge-icon">${icon}</span>
        <span class="risk-badge-label">${label}</span>
    `;
    riskBadgeContainer.appendChild(badge);
}

function renderRiskTags(tags) {
    if (!riskTagsContainer) return;
    riskTagsContainer.innerHTML = "";

    if (!Array.isArray(tags) || tags.length === 0) return;

    tags.forEach((tag) => {
        const span = document.createElement("span");
        span.className = "risk-tag";
        span.textContent = tag;
        riskTagsContainer.appendChild(span);
    });
}

// ============================================
// ENHANCED DOCUMENT RENDERING (UPDATED)
// ============================================

function renderDocuments(documentsData) {
    if (!documentsList) return;
    documentsList.innerHTML = "";

    const docsList = documentsData.required_documents || [];
    const specialInstructions = documentsData.special_instructions || [];

    // Display special instructions if any
    if (docSpecialInstructions && specialInstructions.length > 0) {
        docSpecialInstructions.style.display = 'block';
        docSpecialInstructions.innerHTML = '<strong>⚠️ Special Instructions:</strong><ul>' +
            specialInstructions.map(instr => `<li>${instr}</li>`).join('') +
            '</ul>';
    } else if (docSpecialInstructions) {
        docSpecialInstructions.style.display = 'none';
    }

    if (!Array.isArray(docsList) || docsList.length === 0) {
        const empty = document.createElement("div");
        empty.className = "document-item";
        empty.innerHTML = `
            <div class="document-icon">📂</div>
            <div class="document-main">
                <div class="document-title">No specific documents required</div>
                <div class="document-meta">
                    <span class="document-tag">Standard KYC</span>
                </div>
            </div>
        `;
        documentsList.appendChild(empty);
        return;
    }

    // Categorize documents
    const baseDocKeywords = ['pan', 'aadhaar', 'driving license', 'rc', 'registration'];
    const baseDocs = [];
    const additionalDocs = [];

    docsList.forEach(doc => {
        const docLower = doc.toLowerCase();
        if (baseDocKeywords.some(keyword => docLower.includes(keyword))) {
            baseDocs.push(doc);
        } else {
            additionalDocs.push(doc);
        }
    });

    // Render base documents
    if (baseDocs.length > 0) {
        const header = document.createElement("div");
        header.className = "document-category-header";
        header.innerHTML = `<strong>📋 Base Documents (${baseDocs.length})</strong>`;
        documentsList.appendChild(header);

        baseDocs.forEach(doc => {
            const item = createDocumentItem(doc, 'base');
            documentsList.appendChild(item);
        });
    }

    // Render additional documents
    if (additionalDocs.length > 0) {
        const header = document.createElement("div");
        header.className = "document-category-header";
        header.innerHTML = `<strong>📎 Additional Documents (${additionalDocs.length})</strong>`;
        documentsList.appendChild(header);

        additionalDocs.forEach(doc => {
            const item = createDocumentItem(doc, 'additional');
            documentsList.appendChild(item);
        });
    }
}

function createDocumentItem(docName, category) {
    const item = document.createElement("div");
    item.className = "document-item";

    const icon = category === 'base' ? '📄' : '📎';
    const tag = category === 'base' ? 'Required' : 'Risk-Based';

    item.innerHTML = `
        <div class="document-icon">${icon}</div>
        <div class="document-main">
            <div class="document-title">${docName}</div>
            <div class="document-meta">
                <span class="document-tag document-tag--${category}">${tag}</span>
            </div>
        </div>
    `;

    return item;
}

// ============================================
// ENHANCED PREDICTION UI UPDATE (MAIN FUNCTION)
// ============================================

function updateUIWithPrediction(data, selectedModelName) {
    if (!data) return;

    console.log('📊 Enhanced prediction data:', data);

    // 1) Unpack enhanced backend structure
    const validation = data.validation || {};
    const prediction = data.prediction || {};
    const probs = data.probabilities || {};
    const genai = data.genai_insights || {};
    const docsMeta = data.documents || {};

    // 2) Display validation errors/warnings
    displayValidation(validation);

    // 3) Display red flags
    displayRedFlags(validation);

    // 4) Save context for chat
    lastAssessmentContext = {
        validation: validation,
        prediction: prediction,
        probabilities: probs,
        documents: docsMeta
    };

    // 5) ENHANCED RISK SCORING DISPLAY
    
    // Composite Risk Score (0-100, lower is better)
    const compositeScore = prediction.composite_risk_score;
    if (compositeScoreText && compositeScore !== undefined) {
        compositeScoreText.textContent = compositeScore.toFixed(1);
    }
    
    // Update main score display with composite risk
    if (riskScoreText && compositeScore !== undefined) {
        riskScoreText.textContent = compositeScore.toFixed(0);
    }

    if (scoreFill && compositeScore !== undefined) {
        const clamped = Math.max(0, Math.min(100, compositeScore));
        scoreFill.style.width = clamped + "%";
        
        // Color based on risk level
        if (compositeScore < 30) {
            scoreFill.style.background = '#10b981'; // Green
        } else if (compositeScore < 55) {
            scoreFill.style.background = '#f59e0b'; // Orange
        } else {
            scoreFill.style.background = '#ef4444'; // Red
        }
    }

    // Business Attractiveness (customer interest)
    const businessAttr = prediction.business_attractiveness;
    if (businessAttractivenessText && businessAttr !== undefined) {
        businessAttractivenessText.textContent = businessAttr.toFixed(1) + '/100';
    }

    // Underwriting Risk
    const underwritingRisk = prediction.underwriting_risk;
    if (underwritingRiskText && underwritingRisk !== undefined) {
        underwritingRiskText.textContent = underwritingRisk + '/100';
    }

    // ML Probability (probability customer will buy insurance)
    const mlProb = prediction.ml_probability;
    if (mlProbabilityText && mlProb !== undefined) {
        mlProbabilityText.textContent = safePercent(mlProb);
    }

    // Display risk breakdown chart
    displayRiskBreakdown(prediction);

    // 6) Probability cards
    if (probNotInterested) {
        const notInt = probs.not_interested;
        probNotInterested.textContent = notInt ? safePercent(notInt) : '—';
    }
    if (probInterested) {
        const interested = probs.interested;
        probInterested.textContent = interested ? safePercent(interested) : '—';
    }

    // 7) Risk tier, confidence, action
    const tier = prediction.risk_tier || prediction.risk_level;
    if (tierText) {
        tierText.textContent = safeText(tier);
    }

    if (confidenceText) {
        confidenceText.textContent = safeText(prediction.confidence);
    }

    if (actionText) {
        actionText.textContent = safeText(prediction.action || prediction.recommendation);
    }

    renderRiskBadge(tier);
    renderRiskTags(data.tags || data.risk_tags || []);

    // 8) Meta info
    if (modelUsedText) {
        modelUsedText.textContent = safeText(data.model_used || selectedModelName);
    }

    if (timestampText) {
        const ts = data.timestamp || getCurrentTimestamp();
        timestampText.textContent = safeText(ts);
    }

    // 9) AI insights
    if (aiSummaryText) {
        aiSummaryText.textContent = safeText(genai.summary, "No summary generated.");
    }
    if (aiRationaleText) {
        aiRationaleText.textContent = safeText(genai.rationale, "No rationale provided.");
    }
    if (aiRecommendationText) {
        aiRecommendationText.textContent = safeText(genai.recommendation, "No recommendation generated.");
    }

    // 10) Documents section
    if (docCount) {
        const count = docsMeta.document_count || (docsMeta.required_documents || []).length;
        docCount.textContent = count;
    }
    if (docUrgency) {
        docUrgency.textContent = safeText(docsMeta.urgency);
    }
    if (docReviewType) {
        docReviewType.textContent = safeText(docsMeta.review_type);
    }
    if (docEta) {
        const eta = docsMeta.estimated_processing_days;
        docEta.textContent = eta ? `${eta} days` : '—';
    }

    renderDocuments(docsMeta);

    // 11) Show results
    showResults();
}

// ============================================
// Form submit handler
// ============================================

if (form) {
    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        // Get selected model for display
        const modelSelect = document.getElementById("model");
        const selectedModelName = modelSelect
            ? modelSelect.options[modelSelect.selectedIndex].textContent.trim()
            : null;

        const formData = new FormData(form);

        showLoading();

        const submitButton = form.querySelector('button[type="submit"]');
        const originalBtnText = submitButton ? submitButton.innerHTML : null;
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.innerHTML = `
                <span class="loading-spinner"></span>
                <span>Analyzing...</span>
            `;
        }

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            // Handle validation errors (400)
            if (response.status === 400) {
                const errorData = await response.json();
                
                // Display validation errors
                if (errorData.errors || errorData.warnings) {
                    displayValidation({
                        errors: errorData.errors || [],
                        warnings: errorData.warnings || []
                    });
                }
                
                alert("Validation Error:\n" + (errorData.errors || []).join('\n'));
                return;
            }

            if (!response.ok) {
                throw new Error(
                    `Prediction request failed with status ${response.status}`
                );
            }

            const data = await response.json();
            
            if (!data.success) {
                console.error("Backend error:", data);
                alert(
                    "The backend reported an error:\n" +
                    (data.error || "Unknown error")
                );
                return;
            }

            updateUIWithPrediction(data, selectedModelName);

        } catch (error) {
            console.error("Error during prediction:", error);
            alert(
                "Something went wrong during the risk assessment.\n" +
                "Please check the console and try again."
            );
        } finally {
            hideLoading();
            if (submitButton && originalBtnText !== null) {
                submitButton.disabled = false;
                submitButton.innerHTML = originalBtnText;
            }
        }
    });
}

// ============================================
// Chat panel logic
// ============================================

function toggleChatPanel() {
    if (!chatPanel) return;
    chatPanel.classList.toggle("open");
}

function appendChatMessage(text, sender = "assistant") {
    if (!chatMessages) return;

    const wrapper = document.createElement("div");
    wrapper.className =
        "chat-message " +
        (sender === "user" ? "chat-message-user" : "chat-message-assistant");

    const avatar = document.createElement("div");
    avatar.className = "chat-message-avatar";
    avatar.textContent = sender === "user" ? "You" : "🤖";

    const bubble = document.createElement("div");
    bubble.className = "chat-message-bubble";
    
    // Support markdown-style formatting
    const formattedText = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
    
    bubble.innerHTML = formattedText;

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);

    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendChatMessage() {
    if (!chatInput || !chatInput.value.trim()) return;

    const userText = chatInput.value.trim();
    chatInput.value = "";

    // Show user message
    appendChatMessage(userText, "user");

    // Show typing indicator
    const typingId = 'typing-' + Date.now();
    appendChatMessage("Thinking...", "assistant");
    const typingMsg = chatMessages.lastChild;
    typingMsg.id = typingId;

    try {
        const body = {
            message: userText,
            context: lastAssessmentContext || null
        };

        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(body)
        });

        // Remove typing indicator
        if (typingMsg && typingMsg.id === typingId) {
            typingMsg.remove();
        }

        if (!response.ok) {
            throw new Error(
                `Chat request failed with status ${response.status}`
            );
        }

        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Unknown error');
        }
        
        const reply = data.response || data.reply || data.answer || 
                     "I'm sorry, I couldn't generate a response.";

        appendChatMessage(reply, "assistant");

    } catch (error) {
        console.error("Chat error:", error);
        
        // Remove typing indicator if still there
        if (typingMsg && typingMsg.id === typingId) {
            typingMsg.remove();
        }
        
        appendChatMessage(
            "⚠️ There was an error talking to the AI assistant. Please try again.",
            "assistant"
        );
    }
}

// Handle Enter key in chat input
if (chatInput) {
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
}

// Expose to global scope for onclick handlers in HTML
window.toggleChatPanel = toggleChatPanel;
window.sendChatMessage = sendChatMessage;

// Initialize
console.log('✅ Enhanced Insurance Underwriting Co-Pilot v2.0 loaded');