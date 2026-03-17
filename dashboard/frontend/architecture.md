# GANomics Frontend Architecture

The frontend is a React + TypeScript + Vite application that powers the browser-based dashboard you interact with. It focuses on presenting dataset summaries, training steps, analysis results, and manuscript records while talking to the FastAPI backend.

## Project Layout

```
dashboard/frontend/
├── public/             # static files served by Vite (favicon, metadata)
├── src/                # application source code and assets
│   ├── assets/         # static images, icons, and theme assets
│   ├── components/     # reusable react components grouped by feature
│   │   ├── analysis/   # panels for ablation, sync, DEG, pathway, prediction analytics
│   │   ├── common/     # UI primitives (buttons, cards, typography) centralized in UIComponents
│   │   ├── dashboard/  # project/task/manuscript dashboards and panels
│   │   └── modals/     # dialogs for settings, analytics, and sync flows
│   ├── constants.ts    # shared constants (API base URLs, algorithm labels)
│   ├── types.ts        # TypeScript models shared across components
│   ├── App.tsx         # root layout (header, layout grid, routing placeholders)
│   ├── App.css         # layout-specific styling
│   ├── index.css       # app-wide resets and font imports
│   ├── main.tsx        # Vite entry point that renders `<App />`
│   └── app.tsx.history # history file (can be ignored by AI-assisted edits)
├── package.json        # npm scripts (`dev`, `build`, `lint`, `preview`)
├── tsconfig*.json      # TypeScript configs for Vite tooling and eslint
└── vite.config.ts      # Vite + React plugin configuration
```

## Entry Points & Runtime

- **`src/main.tsx`** loads global CSS/resets and renders `<App />` inside `React.StrictMode`. It also wraps the root in any providers (e.g., `ErrorBoundary` if added later).
- **`src/App.tsx`** defines the main layout (likely a responsive grid) and routes navigation between the project dashboard, analysis area, and manuscript viewer. It orchestrates fetching from `http://localhost:8832/api/...` (matching the FastAPI backend) and passes data to child components.
- **`package.json` scripts**:
  - `npm run dev`: start Vite dev server on http://localhost:5173
  - `npm run build`: produce production bundle
  - `npm run serve`: preview built files via `vite preview`

## Source Code Layers

### Assets & Styling
- `src/assets/`: store logos, icons, or static imagery included in `App.tsx` or component headers.
- `src/App.css` + `src/index.css`: manage layout spacing, background gradients, type scale, and shared tokens. Styles are scoped globally but can be overruled per-component via CSS modules or inline styles.

### Shared Models
- `src/constants.ts`: centralizes algorithm names, color mapping, or endpoint paths, so both `analysis` and `dashboard` components stay consistent.
- `src/types.ts`: defines shared TypeScript interfaces (e.g., `ProjectSummary`, `RunStatus`, `ManuscriptTask`). When adding new endpoints, extend these types and update consuming components.

### Component Tree
- `components/dashboard/`: covers the high-level dashboards:
  - `ProjectDashboard.tsx`: lists datasets, triggers sample uploads, and shows general status cards.
  - `TaskDashboard.tsx`: monitors running tasks/runs and allows manual triggering of analysis steps.
  - `ManuscriptRecords.tsx`: browses `results_ms/` data (tables + downloads).
  - `NewSessionPanel.tsx`: form to launch training/ablation runs (size/beta/lambda selectors).

- `components/analysis/`: houses the most data-rich views. Each file maps to one of the FastAPI steps:
  - `SyncStatusDetails.tsx`, `TsneVisualization.tsx`: sync data, t-SNE export views.
  - `ComparativeAnalysis.tsx`, `DegAnalysis.tsx`, `PathwayAnalysis.tsx`, `PredictionAnalysis.tsx`: show charts/tables from the backend `GET /api/runs/{run_id}/...` endpoints.
  - `AblationCharts.tsx`, `LogViewer.tsx`: training progress and log streaming.

- `components/modals/`: overlays for user actions:
  - `AblationAnalyticsModal.tsx`: quick view of architecture/size/sensitivity logs.
  - `SettingsModal.tsx`: general configuration toggles (API host, dev tooling, etc.).
  - `SyncExternalModal.tsx`: upload external datasets via `POST /api/runs/{run_id}/sync_external`.

- `components/common/UIComponents.tsx`: re-usable UI building blocks such as cards, tables, and buttons. When in doubt, put shared primitives here to avoid duplication.

## Backend Integration

The frontend communicates with FastAPI endpoints via `API_BASE`, which now dynamically resolves to the current page’s origin (`window.location`). That means when the app is served from `http://ncshpcgpu01:8832`, it will automatically point to `http://ncshpcgpu01:8832/api`. To force a different target (e.g., when you build for another host), set the `VITE_API_BASE` environment variable before running `npm run dev` or `npm run build`. The constant file appends `/api` so you only need to supply the host (e.g., `VITE_API_BASE=https://ncshocgpu01:8832`).

When adding new UI routes or data displays:
1. Identify/make the matching backend route in `dashboard/backend/main.py` or `scripts/`.
2. Update `src/types.ts` to reflect the JSON contract.
3. Update constants (if endpoint path, algorithm key, etc. were added).
4. Wire the new view component and import it within `App.tsx` or the relevant dashboard.

## Navigation Tips for AI Strategists

1. **Start from `App.tsx`** to understand the top-level layout and data fetching. It reveals how the UI is split between dashboards, analytics, and modals.
2. **Look up `components/analysis/` for any `run_id`-related display**—each file mirrors a backend step and documents which `/api/runs/{}` endpoint it consumes.
3. **Use `constants.ts` and `types.ts`** to confirm endpoint names, allowed algorithms, or data shapes before editing a component.
4. **When building new features**, follow the flow: backend adds endpoint → `types.ts` + `constants.ts` update → new component in `components/analysis/` or `dashboard/` → `App.tsx`/modal integrates it.
5. **Leverage `components/common/UIComponents.tsx`** to keep a consistent look-and-feel; UI tokens seldom change, so reusing these primitives speeds prototyping.

With this mental map, you can reason about how analytics data flows from the FastAPI backend (`results/`, `results_ms/`), through the scripts, and onto the Vite-powered dashboard without clicking through every file manually.