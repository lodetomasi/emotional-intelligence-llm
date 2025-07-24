"""
Analisi e visualizzazione avanzata dei risultati EI
Crea grafici interattivi e insights dettagliati
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import glob
import os

# Configurazione stile
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_latest_results():
    """Carica i risultati più recenti"""
    json_files = glob.glob("complete_verbose_results_*.json") + \
                 glob.glob("ei_test_results_*.json") + \
                 glob.glob("complete_ei_test_results_*.json")
    
    if not json_files:
        print("❌ Nessun file di risultati trovato!")
        return None
    
    latest_file = max(json_files, key=os.path.getctime)
    print(f"📂 Caricando: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_emotions_from_text(text):
    """Estrae emozioni menzionate nel testo"""
    emotions = [
        'frustration', 'guilt', 'disappointment', 'anger', 'sadness',
        'joy', 'anxiety', 'fear', 'doubt', 'hurt', 'rejection',
        'loneliness', 'stress', 'regret', 'conflict', 'helplessness',
        'resignation', 'worry', 'shame', 'confusion', 'overwhelm'
    ]
    
    text_lower = text.lower()
    found_emotions = []
    
    for emotion in emotions:
        if emotion in text_lower:
            found_emotions.append(emotion)
    
    return found_emotions

def create_emotion_detection_heatmap(results):
    """Crea heatmap delle emozioni rilevate per modello"""
    emotions_data = {}
    
    # Estrai emozioni per ogni modello
    for model_name, model_data in results.get('results', {}).items():
        emotions_found = []
        
        # Cerca nei test di emotion recognition
        for test_name, test_data in model_data.get('tests', {}).items():
            if 'emotion' in test_name.lower() and test_data.get('success'):
                response = test_data.get('response', '')
                emotions_found.extend(extract_emotions_from_text(response))
        
        emotions_data[model_name] = list(set(emotions_found))
    
    # Crea matrice per heatmap
    all_emotions = sorted(set(sum(emotions_data.values(), [])))
    models = list(emotions_data.keys())
    
    matrix = []
    for model in models:
        row = [1 if emotion in emotions_data[model] else 0 for emotion in all_emotions]
        matrix.append(row)
    
    # Crea heatmap interattiva con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=all_emotions,
        y=models,
        colorscale='RdYlBu',
        text=[[f"{'✓' if val else ''}" for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False,
        hovertemplate='Model: %{y}<br>Emotion: %{x}<br>Detected: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Emotion Detection Capabilities by Model',
        xaxis_title='Emotions',
        yaxis_title='Models',
        height=500,
        xaxis={'tickangle': -45}
    )
    
    fig.write_html('emotion_detection_heatmap.html')
    return fig

def create_performance_radar_chart(results):
    """Crea radar chart delle performance per dimensione"""
    dimensions = ['emotion_recognition', 'empathy', 'emotional_regulation', 'social_awareness']
    
    # Calcola success rate per ogni modello e dimensione
    radar_data = []
    
    for model_name, model_data in results.get('results', {}).items():
        scores = []
        for dim in dimensions:
            tests = model_data.get('tests', {})
            
            # Conta successi per questa dimensione
            successes = sum(1 for test_name, test_data in tests.items() 
                          if dim in test_name.lower() and test_data.get('success', False))
            total = sum(1 for test_name in tests.keys() if dim in test_name.lower())
            
            score = (successes / total * 100) if total > 0 else 0
            scores.append(score)
        
        radar_data.append({
            'model': model_name,
            'scores': scores
        })
    
    # Crea radar chart con Plotly
    fig = go.Figure()
    
    for data in radar_data:
        fig.add_trace(go.Scatterpolar(
            r=data['scores'],
            theta=[d.replace('_', ' ').title() for d in dimensions],
            fill='toself',
            name=data['model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Emotional Intelligence Performance by Dimension",
        height=600
    )
    
    fig.write_html('ei_performance_radar.html')
    return fig

def create_response_time_analysis(results):
    """Analizza e visualizza i tempi di risposta"""
    time_data = []
    
    for model_name, model_data in results.get('results', {}).items():
        for test_name, test_data in model_data.get('tests', {}).items():
            if 'time' in test_data:
                time_data.append({
                    'Model': model_name,
                    'Test': test_name.replace('_', ' ').title(),
                    'Response Time (s)': test_data['time'],
                    'Success': test_data.get('success', False)
                })
    
    if not time_data:
        print("⚠️ Nessun dato sui tempi trovato")
        return None
    
    df = pd.DataFrame(time_data)
    
    # Box plot dei tempi di risposta
    fig = px.box(df, x='Model', y='Response Time (s)', 
                 color='Success',
                 title='Response Time Distribution by Model',
                 labels={'Response Time (s)': 'Response Time (seconds)'},
                 color_discrete_map={True: 'green', False: 'red'})
    
    fig.update_layout(height=500)
    fig.write_html('response_time_analysis.html')
    return fig

def create_token_usage_comparison(results):
    """Confronta l'uso dei token tra modelli"""
    token_data = []
    
    for model_name, model_data in results.get('results', {}).items():
        for test_name, test_data in model_data.get('tests', {}).items():
            usage = test_data.get('usage', {})
            if usage and test_data.get('success'):
                token_data.append({
                    'Model': model_name,
                    'Test Type': test_name.split('_')[0].title(),
                    'Prompt Tokens': usage.get('prompt_tokens', 0),
                    'Completion Tokens': usage.get('completion_tokens', 0),
                    'Total Tokens': usage.get('total_tokens', 0)
                })
    
    if not token_data:
        print("⚠️ Nessun dato sui token trovato")
        return None
    
    df = pd.DataFrame(token_data)
    
    # Grouped bar chart
    fig = px.bar(df, x='Model', y='Completion Tokens', 
                 color='Test Type',
                 title='Token Usage by Model and Test Type',
                 barmode='group',
                 text='Completion Tokens')
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(height=500)
    fig.write_html('token_usage_comparison.html')
    return fig

def create_comprehensive_dashboard(results):
    """Crea dashboard completa con subplot"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Success Rate by Model', 'Average Response Length',
                       'Emotion Count by Model', 'Test Completion Rate'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'pie'}]]
    )
    
    models = list(results.get('results', {}).keys())
    
    # 1. Success Rate
    success_rates = []
    for model in models:
        tests = results['results'][model].get('tests', {})
        successes = sum(1 for t in tests.values() if t.get('success', False))
        total = len(tests)
        success_rates.append((successes/total*100) if total > 0 else 0)
    
    fig.add_trace(
        go.Bar(x=models, y=success_rates, name='Success Rate %',
               text=[f'{rate:.1f}%' for rate in success_rates],
               textposition='outside'),
        row=1, col=1
    )
    
    # 2. Average Response Length
    avg_lengths = []
    for model in models:
        tests = results['results'][model].get('tests', {})
        lengths = [len(t.get('response', '')) for t in tests.values() if t.get('success')]
        avg_lengths.append(np.mean(lengths) if lengths else 0)
    
    fig.add_trace(
        go.Bar(x=models, y=avg_lengths, name='Avg Response Length',
               text=[f'{int(length)}' for length in avg_lengths],
               textposition='outside'),
        row=1, col=2
    )
    
    # 3. Emotion Count
    emotion_counts = []
    for model in models:
        tests = results['results'][model].get('tests', {})
        all_emotions = []
        for test_name, test_data in tests.items():
            if 'emotion' in test_name.lower() and test_data.get('success'):
                all_emotions.extend(extract_emotions_from_text(test_data.get('response', '')))
        emotion_counts.append(len(set(all_emotions)))
    
    fig.add_trace(
        go.Scatter(x=models, y=emotion_counts, mode='markers+lines',
                  marker=dict(size=15), name='Unique Emotions',
                  text=[f'{count} emotions' for count in emotion_counts],
                  textposition='top center'),
        row=2, col=1
    )
    
    # 4. Completion Rate Pie
    total_tests = sum(len(m.get('tests', {})) for m in results['results'].values())
    successful_tests = sum(
        sum(1 for t in m.get('tests', {}).values() if t.get('success', False))
        for m in results['results'].values()
    )
    
    fig.add_trace(
        go.Pie(labels=['Successful', 'Failed'], 
               values=[successful_tests, total_tests - successful_tests],
               hole=0.3),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Emotional Intelligence Test - Comprehensive Dashboard",
        showlegend=False,
        height=800
    )
    
    fig.write_html('ei_comprehensive_dashboard.html')
    return fig

def create_emotion_word_frequency(results):
    """Crea visualizzazione della frequenza delle parole emotive"""
    from collections import Counter
    
    emotion_words = []
    
    for model_name, model_data in results.get('results', {}).items():
        for test_name, test_data in model_data.get('tests', {}).items():
            if test_data.get('success'):
                response = test_data.get('response', '').lower()
                # Estrai parole emotive comuni
                for word in ['frustration', 'guilt', 'sadness', 'anxiety', 'fear', 
                           'joy', 'anger', 'disappointment', 'empathy', 'compassion']:
                    emotion_words.extend([word] * response.count(word))
    
    word_freq = Counter(emotion_words)
    
    # Crea bar chart delle parole più frequenti
    words = list(word_freq.keys())[:15]
    counts = [word_freq[w] for w in words]
    
    fig = px.bar(x=counts, y=words, orientation='h',
                 title='Most Frequently Mentioned Emotion Words',
                 labels={'x': 'Frequency', 'y': 'Emotion Words'},
                 color=counts,
                 color_continuous_scale='viridis')
    
    fig.update_layout(height=600)
    fig.write_html('emotion_word_frequency.html')
    return fig

def generate_markdown_report(results):
    """Genera report markdown con insights"""
    report = []
    report.append("# Emotional Intelligence Test Results - Detailed Analysis")
    report.append(f"\n**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Models Tested**: {len(results.get('results', {}))}")
    
    # Model Performance Summary
    report.append("\n## Model Performance Summary\n")
    report.append("| Model | Success Rate | Avg Response Time | Emotions Detected |")
    report.append("|-------|--------------|-------------------|-------------------|")
    
    for model_name, model_data in results.get('results', {}).items():
        tests = model_data.get('tests', {})
        successes = sum(1 for t in tests.values() if t.get('success', False))
        total = len(tests)
        success_rate = (successes/total*100) if total > 0 else 0
        
        avg_time = np.mean([t.get('time', 0) for t in tests.values() if 'time' in t])
        
        emotion_count = len(set(sum([
            extract_emotions_from_text(t.get('response', ''))
            for t in tests.values() if t.get('success')
        ], [])))
        
        report.append(f"| {model_name} | {success_rate:.1f}% | {avg_time:.2f}s | {emotion_count} |")
    
    # Key Insights
    report.append("\n## Key Insights\n")
    
    # Find best performers
    best_success = max(results['results'].items(), 
                      key=lambda x: sum(1 for t in x[1]['tests'].values() if t.get('success', False)))
    report.append(f"- **Most Reliable Model**: {best_success[0]}")
    
    # Sample responses
    report.append("\n## Sample High-Quality Responses\n")
    
    for model_name, model_data in list(results['results'].items())[:3]:
        for test_name, test_data in model_data['tests'].items():
            if test_data.get('success') and 'empathy' in test_name:
                report.append(f"\n### {model_name} - Empathy Response")
                report.append(f"```\n{test_data['response'][:500]}...\n```")
                break
    
    # Save report
    with open('ei_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("📄 Report salvato: ei_analysis_report.md")

def main():
    """Genera tutte le visualizzazioni"""
    print("📊 Generazione analisi e visualizzazioni...")
    
    # Carica risultati
    results = load_latest_results()
    if not results:
        return
    
    print("\n🎨 Creazione grafici...")
    
    # 1. Emotion Detection Heatmap
    print("  1️⃣ Emotion Detection Heatmap...")
    create_emotion_detection_heatmap(results)
    
    # 2. Performance Radar Chart
    print("  2️⃣ Performance Radar Chart...")
    create_performance_radar_chart(results)
    
    # 3. Response Time Analysis
    print("  3️⃣ Response Time Analysis...")
    create_response_time_analysis(results)
    
    # 4. Token Usage Comparison
    print("  4️⃣ Token Usage Comparison...")
    create_token_usage_comparison(results)
    
    # 5. Comprehensive Dashboard
    print("  5️⃣ Comprehensive Dashboard...")
    create_comprehensive_dashboard(results)
    
    # 6. Emotion Word Frequency
    print("  6️⃣ Emotion Word Frequency...")
    create_emotion_word_frequency(results)
    
    # 7. Generate Report
    print("  7️⃣ Generating Markdown Report...")
    generate_markdown_report(results)
    
    print("\n✅ Analisi completata!")
    print("\n📁 File generati:")
    print("  - emotion_detection_heatmap.html")
    print("  - ei_performance_radar.html")
    print("  - response_time_analysis.html")
    print("  - token_usage_comparison.html")
    print("  - ei_comprehensive_dashboard.html")
    print("  - emotion_word_frequency.html")
    print("  - ei_analysis_report.md")
    
    print("\n🌐 Apri i file HTML nel browser per visualizzare i grafici interattivi!")

if __name__ == "__main__":
    main()