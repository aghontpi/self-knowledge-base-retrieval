# Personal Retrieval Assistant — Web UI

A minimal React 19 + TypeScript SPA dashboard to search, monitor, and configure the local retrieval assistant database.

## Styling

Styled using the **[minimal-css-utility](https://github.com/aghontpi/minimal-css-utility)** framework for a lightweight, flexible, and custom presentation.

## Development

```bash
# Install dependencies
pnpm install

# Run the HMR dev server
pnpm run dev
```

The dev server proxies API requests `/api/*` to the FastAPI backend listening on `http://127.0.0.1:8000`.

## Build

```bash
# Compile and bundle static assets into the FastAPI static directory
pnpm run build
```
