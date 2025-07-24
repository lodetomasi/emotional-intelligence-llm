# Security and Methodology Guidelines

## Security Measures Implemented

### 1. API Key Protection
- **Never hardcode API keys** in source files
- Use environment variables or secure configuration files
- API keys are never logged or included in results
- Rotate keys regularly

### 2. File Operation Security
- Path traversal prevention through validation
- Restrictive file permissions (600) on sensitive files
- Output directories restricted to project scope
- Secure filename generation with random suffixes

### 3. Input Validation
- Prompt length limits (max 5000 chars)
- JSON response validation
- URL validation for API endpoints
- Sanitization of user inputs

### 4. Error Handling
- Comprehensive try-catch blocks
- No sensitive data in error messages
- Graceful degradation on failures
- Rate limit handling with exponential backoff

## Methodological Improvements

### 1. Bias Mitigation in Emotion Detection

**Previous Issues:**
- Simple keyword matching favored explicit mentions
- No context consideration
- Binary presence/absence scoring

**Improvements:**
- Contextual pattern matching with regex
- Negation detection ("not happy" vs "happy")
- Confidence scoring (0-1 scale)
- Evidence tracking for transparency

### 2. Empathy Evaluation

**Previous Issues:**
- Keyword counting bias
- Length bias (longer = better)
- No structural analysis

**Improvements:**
- Rubric-based evaluation
- Pattern-based scoring
- Response quality adjustments
- Multiple dimension assessment

### 3. Statistical Validity

**Improvements:**
- No subjective "strength" assessments
- Objective metrics only
- Confidence intervals where applicable
- Clear limitations stated

## Usage Guidelines

### Environment Setup
```bash
# Copy example environment file
cp .env.example .env

# Edit with your API key
# NEVER commit .env to version control
```

### Secure Configuration
```python
# Create secure config
python -c "from ei_improved_framework import create_secure_config; create_secure_config()"
```

### Running Tests Securely
```python
# Use environment variables
export OPENROUTER_API_KEY="your-key"
python run_evaluation.py

# Or use secure config
python run_evaluation.py --config config.secure.json
```

## Best Practices

1. **API Keys**
   - Store in environment variables
   - Use `.env` files locally (never commit)
   - Implement key rotation

2. **Data Handling**
   - Sanitize all outputs
   - No PII in test scenarios
   - Encrypted storage for sensitive results

3. **Evaluation**
   - Always state methodology limitations
   - Include confidence scores
   - Require human validation for critical uses

4. **Reporting**
   - Objective metrics only
   - Clear bias disclaimers
   - Transparent methodology

## Known Limitations

1. **Emotion Detection**
   - Pattern matching has inherent limitations
   - Cultural differences not accounted for
   - Sarcasm and irony detection limited

2. **Empathy Assessment**
   - Structural analysis only
   - Cannot assess genuine understanding
   - Cultural empathy expressions vary

3. **Model Comparison**
   - Different models may interpret prompts differently
   - Response styles vary by training
   - Not all models comparable directly

## Reporting Security Issues

If you discover security vulnerabilities:
1. Do NOT open a public issue
2. Email: lorenzo.detomasi@graduate.univaq.it
3. Include: description, steps to reproduce, impact assessment

## Compliance

This framework follows:
- OWASP Secure Coding Practices
- Academic research ethics guidelines
- Data protection best practices