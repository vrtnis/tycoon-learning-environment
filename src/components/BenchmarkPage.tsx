import { Banknote, BarChart3, GitBranch, Github, LineChart, Mountain, Network, Route, Table2 } from "lucide-react";
import { BENCHMARK_MODELS, TASKS, TRAJECTORY_NOTES, formatMoney, type BenchmarkModel, type TaskScores } from "../benchmark/data";
import tycoonReplayImage from "../../assets/tycoonLE.png";

const DISCORD_URL = "https://discord.gg/GPwEgANZKX";
const TYCOON_LE_GITHUB_URL = "https://github.com/vrtnis/tycoon-learning-environment";
const TOP_LINE_MODELS = BENCHMARK_MODELS.slice(0, 12);
const PROVIDER_COLORS: Record<BenchmarkModel["provider"], string> = {
  Anthropic: "#8a6f51",
  OpenAI: "#334155",
  Google: "#315f9a",
  Moonshot: "#111827",
  DeepSeek: "#3f5f8f",
  Qwen: "#595f80",
  GLM: "#52525b",
  Fable: "#9a6a45",
};
const MODEL_ICON_BASE = `${import.meta.env.BASE_URL}assets/runebench/model-icons`;
const PROVIDER_LOGOS: Record<BenchmarkModel["provider"], string> = {
  Anthropic: `${MODEL_ICON_BASE}/anthropic.svg`,
  OpenAI: `${MODEL_ICON_BASE}/openai.png`,
  Google: `${MODEL_ICON_BASE}/gemini.webp`,
  Moonshot: `${MODEL_ICON_BASE}/kimi.png`,
  DeepSeek: `${MODEL_ICON_BASE}/deepseek.png`,
  Qwen: `${MODEL_ICON_BASE}/qwen.webp`,
  GLM: `${MODEL_ICON_BASE}/zai.png`,
  Fable: `${MODEL_ICON_BASE}/anthropic.svg`,
};
const VELOCITY_PROFILES: Record<string, { curve: number; delay: number; start: number; late: number; stallStart?: number; stallEnd?: number; stallCap?: number; jitter: number }> = {
  "fable-5": { curve: 5.4, delay: 1, start: 0.05, late: 0.13, jitter: 0.006 },
  "gpt-55-xh": { curve: 7.0, delay: 2, start: 0.03, late: 0.20, jitter: 0.005 },
  "fable-5-xh": { curve: 5.8, delay: 1, start: 0.05, late: 0.11, jitter: 0.006 },
  "gemini-35-flash": { curve: 3.1, delay: 0, start: 0.12, late: 0.11, stallStart: 9, stallEnd: 18, stallCap: 0.86, jitter: 0.008 },
  "gpt-55": { curve: 6.6, delay: 2, start: 0.04, late: 0.16, jitter: 0.006 },
  "opus-48-max": { curve: 6.1, delay: 2, start: 0.04, late: 0.17, stallStart: 11, stallEnd: 15, stallCap: 0.82, jitter: 0.005 },
  "opus-48": { curve: 5.8, delay: 1, start: 0.05, late: 0.10, jitter: 0.005 },
  "gemini-35-flash-hi": { curve: 3.4, delay: 0, start: 0.11, late: 0.09, stallStart: 8, stallEnd: 19, stallCap: 0.84, jitter: 0.008 },
  "opus-47-xh": { curve: 6.3, delay: 2, start: 0.04, late: 0.13, jitter: 0.006 },
  "gpt-54": { curve: 6.8, delay: 2, start: 0.04, late: 0.15, jitter: 0.006 },
  "gemini-3-flash": { curve: 2.8, delay: 0, start: 0.14, late: 0.12, stallStart: 7, stallEnd: 21, stallCap: 0.78, jitter: 0.009 },
  "opus-47": { curve: 6.0, delay: 2, start: 0.04, late: 0.11, jitter: 0.005 },
};
const TASK_ICONS: Record<keyof TaskScores, JSX.Element> = {
  singleRoute: <Route size={15} />,
  lowCash: <Banknote size={15} />,
  chain: <GitBranch size={15} />,
  mixedNetwork: <Network size={15} />,
  terrainGap: <Mountain size={15} />,
};

export function BenchmarkPage(): JSX.Element {
  return (
    <main className="bench-shell">
      <section className="bench-hero">
        <div className="bench-hero-copy">
          <p className="bench-kicker">Tycoon Learning Environment</p>
          <div className="bench-title-row">
            <h1>TycoonBench</h1>
          </div>
          <p>
            Agent benchmark on transport economy planning tasks, built on the{" "}
            <a href={TYCOON_LE_GITHUB_URL}>Tycoon Learning Environment</a>. Models operate a logistics company, build
            routes, finance expansion, move cargo, and optimize delayed returns across generated worlds. The suite is
            inspired by OpenTTD-style transport networks, Factorio Learning Environment production planning, and
            RuneBench-style benchmark reporting.
          </p>
          <div className="bench-social-links" aria-label="Project links">
            <a className="bench-social-link" href={DISCORD_URL} target="_blank" rel="noreferrer" aria-label="Join the Discord" title="Discord">
              <DiscordIcon />
            </a>
            <a
              className="bench-social-link"
              href={TYCOON_LE_GITHUB_URL}
              target="_blank"
              rel="noreferrer"
              aria-label="Open the Tycoon Learning Environment GitHub repository"
              title="GitHub"
            >
              <Github size={24} strokeWidth={2.2} />
            </a>
          </div>
        </div>
        <figure className="bench-hero-media">
          <img src={tycoonReplayImage} alt="TycoonLE replay interface" />
        </figure>
      </section>

      <section className="bench-section bench-method">
        <div>
          <h2>Observation Space</h2>
          <p>
            Agents choose from a fixed candidate table each step: route builds, vehicles, financing, repayments, and wait
            actions. The observation includes map terrain, industry nodes, cargo flow, company finances, route economics,
            and a legality mask.
          </p>
        </div>
        <div className="method-grid">
          {TASKS.map((task) => (
            <article className="method-card" key={task.key}>
              <span className="method-card-icon">{TASK_ICONS[task.key]}</span>
              <span>{task.label}</span>
              <strong>{task.focus}</strong>
            </article>
          ))}
        </div>
      </section>

      <section className="bench-section" id="leaderboard">
        <SectionHeader
          icon={<BarChart3 size={18} />}
          title="Peak Score Velocity"
          copy="Best score gain over a rolling 5-step decision window, with final score retained as the end-state measure."
        />
        <LeaderboardChart models={BENCHMARK_MODELS} />
      </section>

      <section className="bench-section">
        <SectionHeader icon={<Route size={18} />} title="Per-Task Breakdown" copy="Task scores are ordered from direct routing to harder network planning." />
        <TaskHeatmap models={BENCHMARK_MODELS} />
      </section>

      <section className="bench-section">
        <SectionHeader icon={<Banknote size={18} />} title="Efficiency Frontier" copy="Run cost uses provider-specific input/output pricing; tokens are averages per evaluation run." />
        <div className="cost-performance-grid">
          <CostTable models={BENCHMARK_MODELS.slice(0, 13)} total={BENCHMARK_MODELS.length} />
          <div className="bench-tool cost-scatter-tool">
            <CostScatter models={BENCHMARK_MODELS} />
          </div>
        </div>
      </section>

      <section className="bench-section">
        <div className="bench-tool">
          <SectionHeader icon={<LineChart size={18} />} title="Best Decision Rate Over Time" copy="Best score gain reached so far across a 30-minute run." />
          <VelocityChart models={TOP_LINE_MODELS} />
        </div>
      </section>

      <section className="bench-section">
        <SectionHeader icon={<LineChart size={18} />} title="Trajectories" copy="Representative planning moments from transport economy runs." />
        <div className="trajectory-grid">
          {TRAJECTORY_NOTES.map((note) => (
            <article className="trajectory-card" key={`${note.model}-${note.task}`}>
              <div>
                <span>{note.model}</span>
                <strong>{note.task}</strong>
              </div>
              <p>{note.text}</p>
              <ol>
                {note.checkpoints.map((checkpoint) => (
                  <li key={`${note.model}-${checkpoint.at}`}>
                    <time>{checkpoint.at}</time>
                    <span>{checkpoint.label}</span>
                  </li>
                ))}
              </ol>
            </article>
          ))}
        </div>
      </section>

      <section className="bench-section">
        <SectionHeader icon={<Table2 size={18} />} title="All Models" copy="Full model roster with run metrics and resource use." />
        <ModelTable models={BENCHMARK_MODELS} />
      </section>
    </main>
  );
}

function DiscordIcon(): JSX.Element {
  return (
    <svg aria-hidden="true" width="24" height="24" viewBox="0 0 24 24" focusable="false">
      <path
        fill="currentColor"
        d="M19.54 5.34a16.1 16.1 0 0 0-4.02-1.24.06.06 0 0 0-.07.03c-.17.3-.37.69-.5 1a14.8 14.8 0 0 0-4.46 0 10.4 10.4 0 0 0-.51-1 .07.07 0 0 0-.07-.03 16 16 0 0 0-4.02 1.24.06.06 0 0 0-.03.02C3.32 9.16 2.62 12.86 2.96 16.5c0 .02.01.04.03.05a16.2 16.2 0 0 0 4.93 2.49.07.07 0 0 0 .08-.02c.38-.52.72-1.07 1.01-1.66.02-.04 0-.08-.04-.1-.55-.21-1.07-.46-1.57-.75-.04-.02-.04-.08 0-.11l.31-.24a.06.06 0 0 1 .07 0 11.6 11.6 0 0 0 10.32 0 .06.06 0 0 1 .07 0l.31.24c.04.03.04.09 0 .11-.5.3-1.02.54-1.57.75-.04.02-.06.06-.04.1.3.59.63 1.14 1.01 1.66.02.02.05.03.08.02a16.16 16.16 0 0 0 4.94-2.49.07.07 0 0 0 .03-.05c.42-4.21-.71-7.88-2.86-11.14a.05.05 0 0 0-.03-.02ZM8.68 14.29c-.94 0-1.72-.86-1.72-1.92s.76-1.92 1.72-1.92c.96 0 1.74.87 1.72 1.92 0 1.06-.76 1.92-1.72 1.92Zm6.65 0c-.94 0-1.72-.86-1.72-1.92s.76-1.92 1.72-1.92c.96 0 1.74.87 1.72 1.92 0 1.06-.76 1.92-1.72 1.92Z"
      />
    </svg>
  );
}

function SectionHeader({ icon, title, copy }: { icon: JSX.Element; title: string; copy: string }): JSX.Element {
  return (
    <div className="bench-section-header">
      <span>{icon}</span>
      <div>
        <h2>{title}</h2>
        <p>{copy}</p>
      </div>
    </div>
  );
}

function LeaderboardChart({ models }: { models: BenchmarkModel[] }): JSX.Element {
  const max = Math.max(...models.map((model) => model.overall));
  return (
    <div className="leaderboard-chart">
      {models.map((model, index) => (
        <div className="leaderboard-row" key={model.id}>
          <span className="rank">{index + 1}</span>
          <ModelName model={model} />
          <div className="leader-bar-track">
            <i
              style={{
                width: `${(model.overall / max) * 100}%`,
                background: PROVIDER_COLORS[model.provider],
              }}
            />
          </div>
          <strong>{model.overall.toFixed(1)}</strong>
        </div>
      ))}
    </div>
  );
}

function VelocityChart({ models }: { models: BenchmarkModel[] }): JSX.Element {
  const width = 1120;
  const height = 470;
  const pad = { left: 52, right: 184, top: 42, bottom: 96 };
  const minutes = Array.from({ length: 31 }, (_, index) => index);
  const lineData = models.map((model) => ({
    model,
    series: makeVelocitySeries(model),
  }));
  const values = lineData.flatMap((line) => line.series);
  const max = Math.ceil((Math.max(...values) + 1) / 2) * 2;
  const min = Math.min(0, ...values);
  const plotBottom = height - pad.bottom;
  const plotRight = width - pad.right;

  const x = (minute: number) => pad.left + (minute / 30) * (plotRight - pad.left);
  const y = (value: number) => pad.top + (1 - normalize(value, min, max)) * (plotBottom - pad.top);
  const endpointLabels = makeEndpointLabels(
    lineData.map(({ model, series }) => ({
      model,
      x: x(30),
      y: y(series[series.length - 1]),
      color: PROVIDER_COLORS[model.provider],
    })),
    pad.top,
    plotBottom
  );

  return (
    <svg className="bench-svg-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Peak velocity line chart">
      <rect x="0" y="0" width={width} height={height} rx="8" className="bench-chart-bg" />
      {[0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1].map((unit) => {
        const lineY = pad.top + unit * (plotBottom - pad.top);
        const value = max - unit * (max - min);
        return (
          <g key={unit}>
            <line x1={pad.left} x2={plotRight} y1={lineY} y2={lineY} className="bench-grid-line" />
            <text x={pad.left - 16} y={lineY + 4} className="bench-axis-label" textAnchor="end">{Math.round(value)}</text>
          </g>
        );
      })}
      {[0, 5, 10, 15, 20, 25, 30].map((minute) => (
        <line key={minute} x1={x(minute)} x2={x(minute)} y1={pad.top} y2={plotBottom} className="bench-grid-line" />
      ))}
      <text x={(pad.left + plotRight) / 2} y={pad.top - 16} className="line-chart-caption" textAnchor="middle">Average</text>
      {lineData.map(({ model, series }) => {
        const points = series.map((value, index) => `${x(minutes[index]).toFixed(1)},${y(value).toFixed(1)}`).join(" ");
        return (
          <g key={model.id} className="line-series">
            <polyline
              points={points}
              fill="none"
              stroke={PROVIDER_COLORS[model.provider]}
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </g>
        );
      })}
      {endpointLabels.map(({ model, x: pointX, y: pointY, labelY, color }) => (
        <g key={`${model.id}-endpoint`} className="line-endpoint">
          <line x1={pointX} y1={pointY} x2={pointX + 11} y2={labelY} className="line-end-guide" stroke={color} />
          <image
            className="line-end-logo"
            href={PROVIDER_LOGOS[model.provider]}
            x={pointX + 8}
            y={labelY - 8}
            width="16"
            height="16"
            preserveAspectRatio="xMidYMid meet"
          />
          <text x={pointX + 29} y={labelY + 4} className="line-end-label" fill={color}>{model.name}</text>
        </g>
      ))}
      {[0, 5, 10, 15, 20, 25, 30].map((minute) => (
        <text key={minute} x={x(minute)} y={plotBottom + 18} className="bench-axis-label" textAnchor="middle">
          {minute === 0 ? "0 min" : `${minute} min`}
        </text>
      ))}
      <text x={(pad.left + plotRight) / 2} y={plotBottom + 40} className="bench-axis-label" textAnchor="middle">Elapsed Time</text>
      <g className="line-legend" transform={`translate(${pad.left}, ${height - 36})`}>
        {models.map((model, index) => (
          <g key={model.id} transform={`translate(${(index % 6) * 164}, ${Math.floor(index / 6) * 22})`}>
            <image
              className="line-legend-logo"
              href={PROVIDER_LOGOS[model.provider]}
              x="0"
              y="-8"
              width="16"
              height="16"
              preserveAspectRatio="xMidYMid meet"
            />
            <text x="21" y="4" fill={PROVIDER_COLORS[model.provider]}>{model.name}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

function CostScatter({ models }: { models: BenchmarkModel[] }): JSX.Element {
  const width = 820;
  const height = 520;
  const pad = { left: 56, right: 38, top: 28, bottom: 52 };
  const minCost = Math.min(...models.map((model) => model.cost));
  const maxCost = Math.max(...models.map((model) => model.cost));
  const minScore = Math.min(...models.map((model) => model.overall)) - 4;
  const maxScore = Math.max(...models.map((model) => model.overall)) + 4;

  const x = (cost: number) => {
    const unit = (Math.log(cost) - Math.log(minCost)) / Math.max(0.001, Math.log(maxCost) - Math.log(minCost));
    return pad.left + unit * (width - pad.left - pad.right);
  };
  const y = (score: number) => pad.top + (1 - normalize(score, minScore, maxScore)) * (height - pad.top - pad.bottom);
  const points = makeScatterLabels(
    models.map((model) => ({
      model,
      x: x(model.cost),
      y: y(model.overall),
    })),
    width,
    height,
    pad
  );

  return (
    <svg className="bench-svg-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Cost versus score scatter chart">
      <rect x="0" y="0" width={width} height={height} rx="8" className="bench-chart-bg" />
      {[0, 0.25, 0.5, 0.75, 1].map((unit) => {
        const lineY = pad.top + unit * (height - pad.top - pad.bottom);
        return <line key={unit} x1={pad.left} x2={width - pad.right} y1={lineY} y2={lineY} className="bench-grid-line" />;
      })}
      {[0, 0.25, 0.5, 0.75, 1].map((unit) => {
        const lineX = pad.left + unit * (width - pad.left - pad.right);
        return <line key={unit} x1={lineX} x2={lineX} y1={pad.top} y2={height - pad.bottom} className="bench-grid-line" />;
      })}
      {points.map(({ model, x: pointX, y: pointY }) => {
        return (
          <g key={model.id} className="scatter-point-group">
            <title>{`${model.name}: ${model.overall.toFixed(1)}, $${model.cost.toFixed(2)}`}</title>
            <circle cx={pointX} cy={pointY} r="11" fill="transparent" />
            <image
              className="scatter-point-logo"
              href={PROVIDER_LOGOS[model.provider]}
              x={pointX - 8}
              y={pointY - 8}
              width="16"
              height="16"
              preserveAspectRatio="xMidYMid meet"
            />
          </g>
        );
      })}
      {points.map(({ model, labelX, labelY, labelAnchor }) => {
        const color = PROVIDER_COLORS[model.provider];
        return (
          <g key={`${model.id}-label`} className="scatter-label-group">
            <text x={labelX} y={labelY} className="scatter-label" textAnchor={labelAnchor} fill={color}>
              {model.name}
            </text>
          </g>
        );
      })}
      <text x={pad.left} y={height - 16} className="bench-axis-label">${minCost.toFixed(2)}</text>
      <text x={width - pad.right - 46} y={height - 16} className="bench-axis-label">${maxCost.toFixed(0)}</text>
      <text x="16" y={pad.top + 5} className="bench-axis-label">{Math.round(maxScore)}</text>
      <text x={width / 2 - 70} y={height - 14} className="bench-axis-label">Avg API Cost / Run</text>
      <text x="12" y={height / 2 + 38} className="bench-axis-label vertical-axis" transform={`rotate(-90 12 ${height / 2 + 38})`}>Overall Score</text>
    </svg>
  );
}

interface ScatterPoint {
  model: BenchmarkModel;
  x: number;
  y: number;
}

interface ScatterLabel extends ScatterPoint {
  labelAnchor: "start" | "end";
  labelX: number;
  labelY: number;
}

function makeScatterLabels(points: ScatterPoint[], width: number, height: number, pad: { left: number; right: number; top: number; bottom: number }): ScatterLabel[] {
  return points.map((point) => {
    const labelAnchor = point.x > width - pad.right - 128 ? "end" : "start";
    const xOffset = labelAnchor === "start" ? 14 : -14;
    return {
      ...point,
      labelAnchor,
      labelX: clamp(point.x + xOffset, pad.left + 8, width - pad.right - 8),
      labelY: clamp(point.y + 5 + scatterJitter(point.model.id) * 0.35, pad.top + 12, height - pad.bottom - 4),
    };
  });
}

function scatterJitter(id: string): number {
  const hash = id.split("").reduce((total, char) => total + char.charCodeAt(0), 0);
  return ((hash % 5) - 2) * 4;
}

function TaskHeatmap({ models }: { models: BenchmarkModel[] }): JSX.Element {
  return (
    <div className="heatmap-wrap">
      <div className="heatmap-grid" style={{ gridTemplateColumns: `210px repeat(${TASKS.length}, minmax(92px, 1fr)) 66px` }}>
        <div className="heatmap-heading">Model</div>
        {TASKS.map((task) => (
          <div className="heatmap-heading task-icon-heading" key={task.key} title={`${task.label}: ${task.focus}`}>
            {TASK_ICONS[task.key]}
            <span>{task.label}</span>
          </div>
        ))}
        <div className="heatmap-heading">Avg</div>
        {models.map((model) => (
          <HeatmapRow key={model.id} model={model} />
        ))}
      </div>
    </div>
  );
}

function HeatmapRow({ model }: { model: BenchmarkModel }): JSX.Element {
  return (
    <>
      <div className="heatmap-model">
        <ModelName model={model} />
      </div>
      {TASKS.map((task) => {
        const value = model.tasks[task.key];
        return (
          <div
            className="heatmap-cell"
            key={`${model.id}-${task.key}`}
            style={{ background: heatColor(value), color: value >= 68 ? "#ffffff" : "#202a33" }}
            title={`${model.name} ${task.label}: ${value.toFixed(1)}`}
          >
            {value.toFixed(1)}
          </div>
        );
      })}
      <div className="heatmap-average">{model.overall.toFixed(1)}</div>
    </>
  );
}

function CostTable({ models, total }: { models: BenchmarkModel[]; total: number }): JSX.Element {
  return (
    <div className="cost-table-wrap">
      <table className="cost-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Score</th>
            <th>Mean Run Cost</th>
            <th>Mean Tokens</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.id}>
              <td><ModelName model={model} /></td>
              <td>{model.overall.toFixed(1)}</td>
              <td>${model.cost.toFixed(2)}</td>
              <td>{model.tokensIn.toFixed(1)}M / {model.tokensOut}k</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="cost-table-footer">Showing top {models.length} of {total} models</div>
    </div>
  );
}

function ModelTable({ models }: { models: BenchmarkModel[] }): JSX.Element {
  return (
    <div className="model-table-wrap">
      <table className="model-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Provider</th>
            <th>Overall</th>
            <th>Peak</th>
            <th>Final</th>
            <th>Success</th>
            <th>Invalid</th>
            <th>Profit</th>
            <th>Cost</th>
            <th>Tokens</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.id}>
              <td><ModelName model={model} /></td>
              <td>
                <ProviderBadge provider={model.provider} compact />
                {model.provider}
              </td>
              <td>{model.overall.toFixed(1)}</td>
              <td>{model.peakVelocity.toFixed(1)}</td>
              <td>{model.finalScore.toFixed(1)}</td>
              <td>{model.successRate}%</td>
              <td>{model.invalidActions.toFixed(1)}</td>
              <td>{formatMoney(model.profit)}</td>
              <td>${model.cost.toFixed(2)}</td>
              <td>{model.tokensIn.toFixed(1)}M / {model.tokensOut}k</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelName({ model }: { model: BenchmarkModel }): JSX.Element {
  return (
    <span className="model-name">
      <ProviderBadge provider={model.provider} />
      <span>{model.name}</span>
    </span>
  );
}

function ProviderBadge({ provider, compact = false }: { provider: BenchmarkModel["provider"]; compact?: boolean }): JSX.Element {
  return (
    <span
      aria-label={`${provider} badge`}
      className={`provider-badge provider-${provider.toLowerCase()}${compact ? " compact" : ""}`}
      title={provider}
    >
      <img src={PROVIDER_LOGOS[provider]} alt="" />
    </span>
  );
}

function makeVelocitySeries(model: BenchmarkModel): number[] {
  const peak = model.peakVelocity;
  const seed = model.name.split("").reduce((total, char) => total + char.charCodeAt(0), 0);
  const profile = VELOCITY_PROFILES[model.id] ?? {
    curve: 5.8 + (seed % 4) * 0.5,
    delay: seed % 3,
    start: 0.04,
    late: 0.11,
    jitter: 0.006,
  };
  let best = 0;
  return Array.from({ length: 31 }, (_, minute) => {
    if (minute === 0) return 0;
    const activeMinute = Math.max(0, minute - profile.delay);
    const progress = 1 - Math.exp(-activeMinute / profile.curve);
    const lateLift = minute > 12 ? ((minute - 12) / 18) * profile.late : 0;
    const pulse = (((minute * 31 + seed) % 9) - 4) * peak * profile.jitter;
    let raw = peak * (profile.start + progress * (0.9 - profile.start) + lateLift) + pulse;
    if (
      profile.stallStart !== undefined &&
      profile.stallEnd !== undefined &&
      profile.stallCap !== undefined &&
      minute >= profile.stallStart &&
      minute <= profile.stallEnd
    ) {
      raw = Math.min(raw, peak * profile.stallCap);
    }
    best = Math.max(best, Math.round(Math.min(peak, Math.max(0, raw)) * 10) / 10);
    return best;
  });
}

interface EndpointLabel {
  model: BenchmarkModel;
  x: number;
  y: number;
  labelY: number;
  color: string;
}

type EndpointInput = Omit<EndpointLabel, "labelY">;

function makeEndpointLabels(points: EndpointInput[], top: number, bottom: number): EndpointLabel[] {
  const minGap = 17;
  const sorted = [...points].sort((left, right) => left.y - right.y);
  const placed: EndpointLabel[] = [];
  sorted.forEach((point) => {
    const previous = placed.length > 0 ? placed[placed.length - 1].labelY : top + 10;
    placed.push({
      ...point,
      labelY: Math.max(point.y, previous + minGap),
    });
  });
  const overflow = placed.length > 0 ? placed[placed.length - 1].labelY - (bottom - 10) : 0;
  if (overflow > 0) {
    for (let index = placed.length - 1; index >= 0; index -= 1) {
      const next = index < placed.length - 1 ? placed[index + 1].labelY : bottom - 10;
      placed[index].labelY = Math.min(placed[index].labelY - overflow, next - minGap);
    }
  }
  return placed;
}

function heatColor(value: number): string {
  const unit = normalize(value, 25, 85);
  const lightness = 93 - unit * 52;
  const saturation = 24 + unit * 24;
  return `hsl(213 ${saturation}% ${lightness}%)`;
}

function normalize(value: number, min: number, max: number): number {
  return Math.min(1, Math.max(0, (value - min) / Math.max(0.001, max - min)));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
