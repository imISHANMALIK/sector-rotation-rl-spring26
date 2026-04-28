export default function Guide() {
  return (
    <div className="space-y-6 max-w-4xl mx-auto pb-10">

      {/* Creators */}
      <div className="card card-glow-blue text-center py-8">
        <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-4">Created by</p>
        <div className="flex flex-col sm:flex-row justify-center gap-6">
          {[
            { name: "Ishan Malik", uni: "im2854" },
            { name: "Maanas Lalwani", uni: "ml10092" },
            { name: "Rishit Maheshwari", uni: "rm7336" },
          ].map(({ name, uni }) => (
            <div key={uni} className="flex flex-col items-center gap-1">
              <span className="text-2xl font-extrabold text-white tracking-tight">{name}</span>
              <span className="text-xs font-mono text-blue-400 tracking-widest uppercase">{uni}</span>
            </div>
          ))}
        </div>
      </div>

      {/* What is this app */}
      <div className="card">
        <h2 className="text-lg font-bold text-white mb-3">What is Lennox Capital?</h2>
        <p className="text-slate-300 text-sm leading-relaxed">
          Lennox Capital is an AI-powered investment dashboard. It uses a{" "}
          <span className="text-blue-400 font-semibold">Reinforcement Learning agent</span> — think of it as a
          robot trader that learned by practicing on years of stock market data — to decide each day which
          sector of the US stock market to invest in. The agent can pick between <span className="text-blue-400 font-semibold">Technology (XLK)</span>,{" "}
          <span className="text-emerald-400 font-semibold">Financials (XLF)</span>,{" "}
          <span className="text-amber-400 font-semibold">Healthcare (XLV)</span>, or simply hold{" "}
          <span className="text-slate-400 font-semibold">Cash</span> when markets look risky.
        </p>
        <p className="text-slate-400 text-sm leading-relaxed mt-3">
          The simulation on the <em>Simulation</em> tab plays back the year 2024 day-by-day and shows you
          exactly what decisions the agent made and how its portfolio performed versus just buying and holding
          the S&P 500 index (SPY).
        </p>
      </div>

      {/* Metric glossary */}
      <div className="card">
        <h2 className="text-lg font-bold text-white mb-4">What Does Each Number Mean?</h2>
        <div className="space-y-5">

          <MetricExplainer
            name="Total Return"
            color="text-emerald-400"
            short="How much money you made (or lost) as a percentage."
            long={`If Total Return is +51%, that means a \$10,000 investment grew to \$15,100 over the year. A negative number means you lost money. Compare this to the SPY Benchmark — if our agent's return is higher, the AI beat the market.`}
            example="Agent: +51.6% vs SPY: +24.9% → the AI outperformed by about 26.7 percentage points."
          />

          <MetricExplainer
            name="Sortino Ratio"
            color="text-blue-400"
            short="How well the returns hold up when you only look at the bad days."
            long={`The Sortino Ratio is like a report card for risk-adjusted performance. It answers: "For every unit of downside risk I took, how much return did I get?" A higher number is better. A ratio above 1.0 is considered good; above 2.0 is excellent. Unlike the more famous Sharpe Ratio, it ignores positive volatility (good days) and only penalises losses.`}
            example="Sortino of 5.25 means the agent earned 5.25 units of return per unit of downside risk — very strong."
          />

          <MetricExplainer
            name="Max Drawdown"
            color="text-red-400"
            short="The worst peak-to-trough loss the portfolio experienced."
            long={`Imagine your portfolio climbed to \$12,000 and then fell to \$10,200 before recovering. That's a drawdown of (\$12,000 − \$10,200) / \$12,000 = 15%. Max Drawdown records the single worst such drop across the whole year. Smaller is better — it tells you how much pain you'd have had to sit through at the worst moment.`}
            example="Max Drawdown of −8.4% means the portfolio never fell more than 8.4% from its highest point."
          />

          <MetricExplainer
            name="SPY Benchmark"
            color="text-purple-400"
            short="What a simple buy-and-hold S&P 500 strategy returned over the same period."
            long={`SPY is an ETF that tracks the 500 largest US companies. It is the standard yardstick every active strategy is measured against. If our agent's Total Return is higher than SPY, the AI added value. If it's lower, a passive index fund would have been better.`}
            example="SPY returned +24.9% in the 2024 holdout period."
          />

          <MetricExplainer
            name="Alpha"
            color="text-emerald-400"
            short="How much extra return the agent earned on top of simply holding SPY."
            long={`Alpha = Agent Return − SPY Return. Positive alpha means the AI beat the market by that much. Professional fund managers are judged heavily on alpha because generating consistent positive alpha is extremely hard.`}
            example="Alpha of +26.7% means the agent outperformed the S&P 500 by 26.7 percentage points over 2024."
          />

          <MetricExplainer
            name="Overrides"
            color="text-amber-400"
            short="How many times a built-in safety rule overruled the AI's pick."
            long={`The agent has a safety guardrail: if market volatility (measured by IV Z-Scores) spikes above a danger threshold, the system automatically moves to Cash regardless of what the AI wanted to do. This protects against extreme market events. Each such intervention is counted as one Override.`}
            example="3 overrides means the safety mechanism kicked in 3 times out of 251 trading days."
          />

          <MetricExplainer
            name="IV Z-Score (Implied Volatility Z-Score)"
            color="text-amber-400"
            short="A measure of how fearful the market is right now, relative to recent history."
            long={`Implied Volatility (IV) reflects how much option traders expect a stock or sector to move. A Z-Score tells you how unusual the current IV is compared to the past 60 days. A Z-Score of 0 is normal. Above +2 means the market is unusually nervous — which is when the override safety rule activates and the agent is forced to hold Cash.`}
            example="Z-Score of 3.1 for XLK means tech volatility is extremely elevated vs. the past two months."
          />

          <MetricExplainer
            name="Q-Values"
            color="text-blue-400"
            short="The AI's confidence score for each possible action it could take."
            long={`Q-values come from inside the neural network. For each possible sector or Cash, the network outputs a number representing its predicted long-term reward from taking that action. The agent picks whichever action has the highest Q-value. Larger spreads between Q-values mean the agent is more decisive; similar Q-values mean it's uncertain.`}
            example="Q: XLK=1.82, XLF=0.93, XLV=0.71, CASH=0.44 → agent strongly prefers Technology."
          />

          <MetricExplainer
            name="Realized Volatility"
            color="text-slate-300"
            short="How much a sector actually moved in recent weeks (backwards-looking)."
            long={`While IV is forward-looking (fear about the future), Realized Volatility (RV) is backward-looking — it measures the actual daily price swings over the past 20 trading days. High RV means the sector has been choppy lately. The agent uses RV as part of its 9-feature input to understand current market conditions.`}
            example="RV of 28% for XLK means tech stocks moved roughly 28% annualized over the past month."
          />

        </div>
      </div>

      {/* How to interpret results */}
      <div className="card">
        <h2 className="text-lg font-bold text-white mb-4">How to Read the Simulation Results</h2>
        <ol className="space-y-4 text-sm text-slate-300 list-none">

          <Step n={1} title="Start the simulation">
            Go to the <span className="text-blue-400 font-semibold">Simulation</span> tab and press{" "}
            <span className="font-mono bg-slate-800 px-1.5 py-0.5 rounded text-white text-xs">Run Simulation</span>.
            The dashboard will replay every trading day of 2024 in real time. You can speed it up using the
            speed buttons.
          </Step>

          <Step n={2} title="Watch the Equity Curve">
            The top chart shows two lines: the <span className="text-blue-400 font-semibold">blue line (Agent)</span>{" "}
            and the <span className="text-purple-400 font-semibold">purple line (SPY)</span>. If the blue line
            climbs faster and stays higher, the AI is beating the market. A blue line that dips sharply below
            its recent peak is a drawdown event.
          </Step>

          <Step n={3} title="Check the metric cards">
            The six cards at the top update live. Focus first on{" "}
            <span className="text-emerald-400 font-semibold">Total Return vs SPY</span> — that tells you
            whether the strategy added value at a glance. Then look at{" "}
            <span className="text-blue-400 font-semibold">Sortino Ratio</span>: anything above 1.5 in a real
            strategy would be considered strong.
          </Step>

          <Step n={4} title="Look at the Sector Pie Chart">
            The pie shows where the agent spent most of its time. A large{" "}
            <span className="text-slate-400 font-semibold">Cash</span> slice is not necessarily bad — it means
            the AI was cautious. A large{" "}
            <span className="text-blue-400 font-semibold">XLK (Technology)</span> slice means it strongly
            favored tech, which historically has high returns but also high risk.
          </Step>

          <Step n={5} title="Spot overrides in the trade log">
            Any row in the Trade Log with a{" "}
            <span className="font-mono bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded text-xs">OVERRIDE</span>{" "}
            badge means the safety system moved to Cash that day because volatility was too high. These are the
            days the agent was protected from potential crashes.
          </Step>

          <Step n={6} title="What a 'good' result looks like">
            A strategy performing well will show: Total Return significantly above SPY, Sortino Ratio above 2,
            Max Drawdown smaller than SPY's, and a positive Alpha. The benchmark for 2024 is SPY ≈ +25%.
            If the agent returns more than that with a Sortino above 2, it has demonstrated genuine skill.
          </Step>

        </ol>
      </div>

      {/* Disclaimer */}
      <p className="text-center text-xs text-slate-600 px-4">
        This dashboard is a research prototype built for educational purposes. Past performance of backtested
        strategies does not guarantee future results. Nothing here constitutes financial advice.
      </p>

    </div>
  );
}

function MetricExplainer({
  name, color, short, long, example,
}: {
  name: string; color: string; short: string; long: string; example: string;
}) {
  return (
    <div className="border border-border rounded-lg p-4 space-y-2">
      <h3 className={`font-bold text-base ${color}`}>{name}</h3>
      <p className="text-white text-sm font-medium">{short}</p>
      <p className="text-slate-400 text-sm leading-relaxed">{long}</p>
      <p className="text-xs text-slate-500 bg-slate-800/60 rounded px-3 py-1.5 font-mono">
        <span className="text-slate-400 font-semibold not-italic">Example: </span>{example}
      </p>
    </div>
  );
}

function Step({
  n, title, children,
}: {
  n: number; title: string; children: React.ReactNode;
}) {
  return (
    <li className="flex gap-4">
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-blue-500/20 border border-blue-500/40 flex items-center justify-center text-blue-400 font-bold text-xs">
        {n}
      </div>
      <div>
        <p className="font-semibold text-white mb-0.5">{title}</p>
        <p className="text-slate-400 leading-relaxed">{children}</p>
      </div>
    </li>
  );
}
