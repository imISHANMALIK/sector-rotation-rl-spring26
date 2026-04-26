"""
app.py
======
Streamlit demo app for the Sector Rotation RL agent.

HOW TO RUN:
    streamlit run demo/app.py

TABS:
1. Live Trading  — step through 2024 data, watch agent decide
2. Performance   — full metrics and charts vs SPY
3. Risk Monitor  — IV z-scores and override trigger history
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.dqn_agent   import DQNAgent
from src.environment import SectorRotationEnv
from src.evaluate    import (sortino_ratio, sharpe_ratio,
                              max_drawdown, total_return,
                              cumulative_returns)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title='Sector Rotation RL',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'iv_features.csv')
CKPT_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'best_model.pt')
CFG_PATH  = os.path.join(ROOT_DIR, 'configs', 'hyperparams.yaml')

COLORS = {
    'XLK':  '#2196F3',
    'XLF':  '#4CAF50',
    'XLV':  '#FF9800',
    'CASH': '#9E9E9E',
    'SPY':  '#FF5722',
}


@st.cache_resource
def load_agent():
    """Load trained DQN agent — cached so it only loads once."""
    if not os.path.exists(CKPT_PATH):
        return None
    with open(CFG_PATH) as f:
        config = yaml.safe_load(f)
    agent = DQNAgent(
        state_dim  = config['state_dim'],
        action_dim = config['action_dim'],
        hidden     = config['hidden_dim'],
    )
    agent.load(CKPT_PATH)
    agent.epsilon = 0.0
    return agent


@st.cache_data
def load_data():
    """Load and cache feature dataset."""
    return pd.read_csv(DATA_PATH, parse_dates=['date'])


@st.cache_data
def load_backtest_results():
    """Load pre-computed backtest results if available."""
    path = os.path.join(ROOT_DIR, 'notebooks', 'backtest_results.csv')
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['date'])
    return None


def render_sidebar():
    st.sidebar.title('⚙️ Controls')
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Model Status')

    if os.path.exists(CKPT_PATH):
        st.sidebar.success('✅ best_model.pt loaded')
    else:
        st.sidebar.error('❌ No checkpoint found')
        st.sidebar.code('python src/train.py')

    st.sidebar.markdown('---')
    st.sidebar.markdown('### Override Settings')
    threshold = st.sidebar.slider(
        'Z-Score Threshold', 1.0, 4.0, 2.5, 0.1,
        help='Override triggers when ALL sector z-scores exceed this'
    )

    st.sidebar.markdown('---')
    st.sidebar.markdown("""
    ### About
    **Risk-Aware RL Sector Rotation**
    - Agent: Deep Q-Network (DQN)
    - State: 9-dim IV feature vector
    - Actions: XLK | XLF | XLV | CASH
    - Reward: Sortino ratio shaped
    - Safety: Vasant Dhar Override

    **Team Lennox | DS-GA 3001 | NYU**
    """)
    return threshold


def render_live_trading(agent, df):
    st.header('▶ Live Trading Simulation')
    st.markdown(
        'Watch the agent make trading decisions day by day through 2024.'
    )

    test_df = df[df['date'] >= '2024-01-01'].reset_index(drop=True)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f'📅 Test period: 2024 ({len(test_df)} trading days)')
    with col2:
        st.info('🎯 Agent always uses greedy policy (ε=0) during demo')

    if st.button('▶ Run Full Simulation', type='primary'):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        env      = SectorRotationEnv(DATA_PATH, mode='test')
        state, _ = env.reset()

        records         = []
        portfolio_value = 1.0
        action_names    = {0: 'XLK', 1: 'XLF', 2: 'XLV', 3: 'CASH'}
        step            = 0

        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)

            portfolio_value *= np.exp(info['daily_return'])
            records.append({
                'date':            info['date'],
                'action':          action_names[info['action_executed']],
                'override':        info['override_triggered'],
                'daily_return':    info['daily_return'],
                'portfolio_value': portfolio_value,
            })

            step += 1
            progress_bar.progress(min(step / len(test_df), 1.0))
            status_text.text(
                f"Day {step}/{len(test_df)} | "
                f"{info['date']} | "
                f"Action: {action_names[info['action_executed']]} | "
                f"Portfolio: ${portfolio_value:.4f}"
            )

            state = next_state
            if terminated or truncated:
                break

        res_df = pd.DataFrame(records)
        res_df['date'] = pd.to_datetime(res_df['date'])
        agent_rets     = res_df['daily_return'].values
        spy_rets       = test_df['ret_spy'].dropna().values

        final_ret = (portfolio_value - 1) * 100
        st.success(
            f'✅ Simulation complete! '
            f'Final value: ${portfolio_value:.4f} ({final_ret:+.1f}%)'
        )

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Sortino', f'{sortino_ratio(agent_rets):.4f}',
                    f'{sortino_ratio(agent_rets)-sortino_ratio(spy_rets):+.4f} vs SPY')
        col2.metric('Total Return', f'{total_return(agent_rets):+.2f}%',
                    f'{total_return(agent_rets)-total_return(spy_rets):+.2f}% vs SPY')
        col3.metric('Max Drawdown', f'{max_drawdown(agent_rets)*100:.2f}%',
                    f'{(max_drawdown(agent_rets)-max_drawdown(spy_rets))*100:+.2f}% vs SPY')
        col4.metric('Override Triggers', f'{int(res_df["override"].sum())}')

        col1, col2 = st.columns(2)

        with col1:
            # Cumulative returns
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=res_df['date'], y=res_df['portfolio_value'],
                name=f'DQN Agent ({total_return(agent_rets):+.1f}%)',
                line=dict(color='#2196F3', width=2.5)
            ))
            spy_cum = np.exp(np.cumsum(spy_rets))
            fig.add_trace(go.Scatter(
                x=test_df.dropna(subset=['ret_spy'])['date'].values,
                y=spy_cum,
                name=f'SPY ({total_return(spy_rets):+.1f}%)',
                line=dict(color='#FF5722', width=2, dash='dash')
            ))
            override_rows = res_df[res_df['override']]
            if len(override_rows) > 0:
                fig.add_trace(go.Scatter(
                    x=override_rows['date'],
                    y=override_rows['portfolio_value'],
                    mode='markers', name='Override (CASH)',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
            fig.update_layout(
                title='Cumulative Portfolio Value',
                height=380, xaxis_title='Date',
                yaxis_title='Value'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Pie chart
            ac = res_df['action'].value_counts()
            fig = px.pie(
                values=ac.values, names=ac.index,
                title='Sector Allocation (2024)',
                color=ac.index, color_discrete_map=COLORS
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        # Trading log
        st.subheader('📋 Trading Log (last 20 days)')
        display = res_df.tail(20).copy()
        display['daily_return']    = display['daily_return'].map('{:+.4f}'.format)
        display['portfolio_value'] = display['portfolio_value'].map('{:.4f}'.format)
        st.dataframe(display, use_container_width=True)


def render_performance(df):
    st.header('📊 Performance Dashboard')

    results_df = load_backtest_results()

    if results_df is None:
        st.warning(
            '⚠️ No backtest results found. Run first:\n'
            '```python -m src.backtest --checkpoint checkpoints/best_model.pt```'
        )
        return

    test_df    = df[df['date'] >= '2024-01-01'].reset_index(drop=True)
    agent_rets = results_df['daily_return'].values
    spy_rets   = test_df['ret_spy'].dropna().values

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric('Sortino', f'{sortino_ratio(agent_rets):.4f}',
                f'SPY: {sortino_ratio(spy_rets):.4f}')
    col2.metric('Sharpe', f'{sharpe_ratio(agent_rets):.4f}',
                f'SPY: {sharpe_ratio(spy_rets):.4f}')
    col3.metric('Total Return', f'{total_return(agent_rets):+.2f}%',
                f'SPY: {total_return(spy_rets):+.2f}%')
    col4.metric('Max Drawdown', f'{max_drawdown(agent_rets)*100:.2f}%',
                f'SPY: {max_drawdown(spy_rets)*100:.2f}%')
    col5.metric('Overrides', f'{int(results_df["override"].sum())}')

    st.markdown('---')
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['date'], y=results_df['portfolio_value'],
            name='DQN Agent', line=dict(color='#2196F3', width=2.5),
            fill='tozeroy', fillcolor='rgba(33,150,243,0.1)'
        ))
        spy_cum = np.exp(np.cumsum(spy_rets))
        fig.add_trace(go.Scatter(
            x=test_df.dropna(subset=['ret_spy'])['date'].values,
            y=spy_cum, name='SPY',
            line=dict(color='#FF5722', width=2, dash='dash')
        ))
        fig.update_layout(
            title='Cumulative Returns (2024)',
            height=350, xaxis_title='Date',
            yaxis_title='Portfolio Value'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Drawdown
        ac     = results_df['portfolio_value'].values
        ap     = np.maximum.accumulate(ac)
        add    = (ac - ap) / ap * 100
        sc     = np.exp(np.cumsum(spy_rets))
        sp     = np.maximum.accumulate(sc)
        sdd    = (sc - sp) / sp * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['date'], y=add, name='DQN Agent',
            line=dict(color='#2196F3'),
            fill='tozeroy', fillcolor='rgba(33,150,243,0.2)'
        ))
        fig.add_trace(go.Scatter(
            x=test_df.dropna(subset=['ret_spy'])['date'].values,
            y=sdd, name='SPY',
            line=dict(color='#FF5722', dash='dash'),
            fill='tozeroy', fillcolor='rgba(255,87,34,0.1)'
        ))
        fig.update_layout(
            title='Drawdown (%)', height=350,
            xaxis_title='Date', yaxis_title='Drawdown %'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Sector allocation
    col_name = ('action_executed'
                if 'action_executed' in results_df.columns
                else 'action')
    if col_name in results_df.columns:
        st.subheader('Sector Allocation')
        ac = results_df[col_name].value_counts()
        fig = px.bar(
            x=ac.index, y=ac.values,
            color=ac.index, color_discrete_map=COLORS,
            text=[f'{v}d ({v/len(results_df)*100:.1f}%)' for v in ac.values],
            labels={'x': 'Sector', 'y': 'Days'},
            title='Agent Sector Allocation (2024 Test Period)'
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_risk_monitor(df, threshold):
    st.header('🛡️ Risk Monitor — Vasant Dhar Override')

    st.markdown(f"""
    The **Vasant Dhar Override** forces the agent to CASH when
    **all three sector IV z-scores simultaneously exceed {threshold:.1f}**.

    This implements the *Automation Frontier* principle: when markets
    enter extreme regimes the agent hasn't trained on, defer to safety.
    The override is **hard-coded outside the gradient pathway** —
    the agent cannot learn to ignore it.
    """)

    zscore_cols   = ['zscore_xlk', 'zscore_xlf', 'zscore_xlv']
    zcols_present = [c for c in zscore_cols if c in df.columns]

    if zcols_present:
        override_mask = (df[zcols_present] > threshold).all(axis=1)

        col1, col2, col3 = st.columns(3)
        col1.metric('Total Override Days', f'{override_mask.sum()}',
                    f'{override_mask.mean()*100:.1f}% of all days')
        col2.metric('First Trigger',
                    str(df[override_mask]['date'].iloc[0])[:10]
                    if override_mask.sum() > 0 else 'None')
        col3.metric('Last Trigger',
                    str(df[override_mask]['date'].iloc[-1])[:10]
                    if override_mask.sum() > 0 else 'None')

        # Z-score time series
        fig = go.Figure()
        colors_z = {
            'zscore_xlk': '#2196F3',
            'zscore_xlf': '#4CAF50',
            'zscore_xlv': '#FF9800'
        }
        labels_z = {
            'zscore_xlk': 'XLK z-score',
            'zscore_xlf': 'XLF z-score',
            'zscore_xlv': 'XLV z-score'
        }

        for col in zcols_present:
            fig.add_trace(go.Scatter(
                x=df['date'], y=df[col],
                name=labels_z.get(col, col),
                line=dict(color=colors_z.get(col, 'gray'), width=1),
                opacity=0.8
            ))

        fig.add_hline(
            y=threshold, line_dash='dash',
            line_color='red', line_width=2,
            annotation_text=f'Override threshold ({threshold})'
        )

        # Shade override periods
        if override_mask.sum() > 0:
            for d in df[override_mask]['date']:
                fig.add_vrect(
                    x0=d,
                    x1=d + pd.Timedelta(days=1),
                    fillcolor='red', opacity=0.3,
                    line_width=0
                )

        fig.update_layout(
            title='IV Z-Scores (Red = Override Triggered)',
            height=420,
            xaxis_title='Date',
            yaxis_title='Z-Score'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Override dates table
        if override_mask.sum() > 0:
            st.subheader('Override Trigger Dates')
            ov_df = df[override_mask][['date'] + zcols_present].copy()
            ov_df['date'] = ov_df['date'].dt.strftime('%Y-%m-%d')
            ov_df.columns = ['Date'] + [
                c.replace('zscore_', '').upper() + ' z-score'
                for c in zcols_present
            ]
            st.dataframe(ov_df, use_container_width=True)
    else:
        st.warning('Z-score columns not found in dataset.')


def main():
    st.title('📈 Risk-Aware Sector Rotation RL')
    st.markdown(
        '**DQN agent rotating between XLK / XLF / XLV using Implied '
        'Volatility signals with a Vasant Dhar safety override**'
    )
    st.markdown('---')

    agent     = load_agent()
    df        = load_data()
    threshold = render_sidebar()

    if agent is None:
        st.error('⚠️ No trained model found.')
        st.code('python src/train.py')
        st.stop()

    tab1, tab2, tab3 = st.tabs([
        '▶ Live Trading',
        '📊 Performance Dashboard',
        '🛡️ Risk Monitor'
    ])

    with tab1:
        render_live_trading(agent, df)

    with tab2:
        render_performance(df)

    with tab3:
        render_risk_monitor(df, threshold)


if __name__ == '__main__':
    main()