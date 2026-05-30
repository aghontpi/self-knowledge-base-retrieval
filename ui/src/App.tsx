import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  RefreshCw, 
  Terminal, 
  FileText, 
  Code, 
  Layers, 
  Sliders, 
  Copy, 
  Check, 
  ExternalLink, 
  X,
  FileCode,
  FolderOpen
} from 'lucide-react';

interface StatsData {
  db_path: string;
  data_dir: string;
  prose: {
    collection: string;
    model: string;
    chunks: number;
  };
  code: {
    collection: string;
    model: string;
    chunks: number;
  };
  reranker: {
    enabled: boolean;
    model: string;
  };
}

interface SearchHit {
  doc_id: string;
  locator: string;
  text: string;
  score: number;
  chunk_index: number;
  source: string;
  rerank_score: number | null;
}

interface SearchResponse {
  hits: SearchHit[];
  grouped?: {
    prose: SearchHit[];
    code: SearchHit[];
  };
}

export default function App() {
  // Database Stats
  const [stats, setStats] = useState<StatsData | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState<string | null>(null);

  // Search Controls
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [useRerank, setUseRerank] = useState(true);
  const [displayMode, setDisplayMode] = useState<'unified' | 'grouped'>('unified');
  
  // Search Results
  const [searchLoading, setSearchLoading] = useState(false);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Ingestion logs
  const [ingesting, setIngesting] = useState(false);
  const [ingestLogs, setIngestLogs] = useState<string[]>([]);
  const [ingestSummary, setIngestSummary] = useState<string | null>(null);
  const logTerminalEndRef = useRef<HTMLDivElement>(null);

  // Active Document Preview
  const [previewDocId, setPreviewDocId] = useState<string | null>(null);
  const [previewContent, setPreviewContent] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  // Copy helpers
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Fetch diagnostics stats
  const fetchStats = async () => {
    setStatsLoading(true);
    setStatsError(null);
    try {
      const res = await fetch('/api/stats');
      if (!res.ok) {
        throw new Error(`Error: ${res.statusText}`);
      }
      const data = await res.json();
      setStats(data);
    } catch (err: any) {
      setStatsError(err.message || 'Failed to fetch statistics');
    } finally {
      setStatsLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  // Scroll terminal logs to bottom when they update
  useEffect(() => {
    if (logTerminalEndRef.current) {
      logTerminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [ingestLogs]);

  // Handle Search Submission
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setSearchLoading(true);
    setSearchError(null);
    setResults(null);

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          top_k: topK,
          use_rerank: useRerank,
          mode: displayMode
        })
      });

      if (!res.ok) {
        throw new Error(`Search failed: ${res.statusText}`);
      }

      const data = await res.json();
      setResults(data);
    } catch (err: any) {
      setSearchError(err.message || 'Search failed');
    } finally {
      setSearchLoading(false);
    }
  };

  // Run database ingestion pipeline
  const runIngest = async () => {
    if (ingesting) return;
    setIngesting(true);
    setIngestSummary(null);
    setIngestLogs(['[ingest] Starting convergent filesystem scan...', '[ingest] Locking vector database...']);

    try {
      const res = await fetch('/api/ingest', { method: 'POST' });
      if (!res.ok) {
        throw new Error(`Ingest failed: ${res.statusText}`);
      }
      const data = await res.json();
      
      setIngestLogs(prev => [
        ...prev,
        '[ingest] Database write-lock acquired.',
        '[ingest] Calculating file integrity hashes...',
        ...data.details.map((d: string) => `[ingest] ${d}`),
        '[ingest] Synchronization completed successfully.'
      ]);
      setIngestSummary(data.summary);
      
      // Refresh metrics
      fetchStats();
    } catch (err: any) {
      setIngestLogs(prev => [...prev, `[ERROR] Ingestion failed: ${err.message}`]);
    } finally {
      setIngesting(false);
    }
  };

  // Open safe file preview
  const openFilePreview = async (docId: string) => {
    setPreviewDocId(docId);
    setPreviewContent(null);
    setPreviewLoading(true);
    setPreviewError(null);

    try {
      const res = await fetch(`/api/file?doc_id=${encodeURIComponent(docId)}`);
      if (!res.ok) {
        throw new Error(`Failed to load file: ${res.statusText}`);
      }
      const data = await res.json();
      if (data.error) {
        throw new Error(data.error);
      }
      setPreviewContent(data.content);
    } catch (err: any) {
      setPreviewError(err.message || 'Failed to inspect document');
    } finally {
      setPreviewLoading(false);
    }
  };

  // Copy text helper
  const handleCopyText = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  const getFileIcon = (docId: string) => {
    const ext = docId.split('.').pop()?.toLowerCase();
    if (['py', 'js', 'ts', 'tsx', 'cpp', 'rs', 'go', 'json', 'yaml', 'toml', 'sh'].includes(ext || '')) {
      return <Code size={15} style={{ color: 'var(--text-secondary)' }} />;
    }
    return <FileText size={15} style={{ color: 'var(--text-secondary)' }} />;
  };

  const getScoreTag = (hit: SearchHit) => {
    if (useRerank && hit.rerank_score !== null) {
      return (
        <span className="badge badge-score">
          Rerank: {hit.rerank_score.toFixed(4)}
        </span>
      );
    }
    return (
      <span className="badge badge-score">
        Cos: {hit.score.toFixed(4)}
      </span>
    );
  };

  return (
    <div className="dashboard-grid">

      {/* LEFT PANEL — DIAGNOSTICS & SYSTEM CONTROL */}
      <aside style={{
        backgroundColor: 'var(--bg-primary)',
        borderRight: '1px solid var(--border-color)',
        padding: '1.5rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.5rem',
        overflowY: 'auto'
      }}>
        {/* Simple Text Branding Header (No Logo Element) */}
        <div>
          <h1 style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
            PRA Console
          </h1>
          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
            Local Retrieval Assistant
          </span>
        </div>

        <hr className="separator" />

        {/* Database Sync Control */}
        <div className="flat-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '0.85rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Terminal size={14} style={{ color: 'var(--text-secondary)' }} />
              Indexing Pipeline
            </h3>
            {ingesting && <span className="spinner" style={{ color: 'var(--text-secondary)' }}></span>}
          </div>
          
          <button 
            onClick={runIngest} 
            disabled={ingesting}
            className="btn-minimal-dark" 
            style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', padding: '0.5rem 1rem' }}
          >
            <RefreshCw size={13} className={ingesting ? 'spinner' : ''} />
            {ingesting ? 'Syncing...' : 'Sync Index'}
          </button>

          {ingestLogs.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Integrity log stream:</span>
              <div className="terminal-view" style={{ height: '110px', fontSize: '0.725rem', padding: '0.5rem 0.75rem' }}>
                {ingestLogs.map((log, index) => (
                  <div key={index} style={{ marginBottom: '0.2rem' }}>{log}</div>
                ))}
                <div ref={logTerminalEndRef} />
              </div>
            </div>
          )}

          {ingestSummary && (
            <div style={{ 
              padding: '0.6rem', 
              borderRadius: '6px', 
              backgroundColor: '#f0fdf4', 
              border: '1px solid #bbf7d0',
              fontSize: '0.725rem', 
              color: 'var(--accent-emerald)',
              lineHeight: '1.4'
            }}>
              {ingestSummary}
            </div>
          )}
        </div>

        {/* Database Diagnostic Metrics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
          <h3 style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
            System Diagnostics
          </h3>
          
          {statsLoading ? (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '1rem' }}>
              <span className="spinner" style={{ color: 'var(--text-muted)' }}></span>
            </div>
          ) : statsError ? (
            <div style={{ fontSize: '0.725rem', color: 'var(--accent-rose)', padding: '0.5rem', backgroundColor: '#fef2f2', border: '1px solid #fee2e2', borderRadius: '6px' }}>
              {statsError}
            </div>
          ) : stats ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              {/* Database path */}
              <div className="flat-card" style={{ padding: '0.75rem', fontSize: '0.725rem', display: 'flex', flexDirection: 'column', gap: '0.2' }}>
                <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  DB File:
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', wordBreak: 'break-all' }}>
                  {stats.db_path.split('/').pop()}
                </span>
              </div>

              {/* Data directory */}
              <div className="flat-card" style={{ padding: '0.75rem', fontSize: '0.725rem', display: 'flex', flexDirection: 'column', gap: '0.2' }}>
                <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <FolderOpen size={11} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '3px' }} /> Data Dir:
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', wordBreak: 'break-all' }}>
                  {stats.data_dir}
                </span>
              </div>

              {/* Prose stats card */}
              <div className="flat-card" style={{ padding: '0.75rem', borderLeft: '3px solid var(--accent-amber)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                  <span className="badge badge-prose">Prose Index</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>{stats.prose.chunks}</span>
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column' }}>
                  <span>Collection: {stats.prose.collection}</span>
                  <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }} title={stats.prose.model}>
                    Model: {stats.prose.model.split('/').pop()}
                  </span>
                </div>
              </div>

              {/* Code stats card */}
              <div className="flat-card" style={{ padding: '0.75rem', borderLeft: '3px solid var(--accent-emerald)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                  <span className="badge badge-code">Code Index</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>{stats.code.chunks}</span>
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column' }}>
                  <span>Collection: {stats.code.collection}</span>
                  <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }} title={stats.code.model}>
                    Model: {stats.code.model.split('/').pop()}
                  </span>
                </div>
              </div>

              {/* Reranker Card */}
              <div className="flat-card" style={{ padding: '0.75rem', borderLeft: '3px solid #64748b' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                  <span className="badge badge-score">Rerank stage</span>
                  <span style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    {stats.reranker.enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
                {stats.reranker.enabled && (
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>
                    <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', display: 'block' }} title={stats.reranker.model}>
                      Model: {stats.reranker.model.split('/').pop()}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>

        <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '0.2rem', fontSize: '0.65rem', color: 'var(--text-muted)', textAlign: 'center' }}>
          <span>Personal Retrieval Assistant</span>
          <span>MIT License © 2026</span>
        </div>
      </aside>

      {/* MAIN CONTAINER — SEMANTIC SEARCH ENGINE */}
      <main style={{
        padding: '2rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.5rem',
        overflowY: 'auto'
      }}>
        {/* Search Console Header */}
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Semantic Search Console
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
            Query prose notes and codebase syntax together across models in a lightweight workspace.
          </p>
        </div>

        {/* Search Input Bar Card */}
        <form onSubmit={handleSearch} className="flat-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="clean-input-container">
            <Search className="search-icon-inside" size={18} />
            <input 
              type="text" 
              placeholder="e.g. how is retry logic implemented in python?" 
              value={query} 
              onChange={e => setQuery(e.target.value)}
              className="clean-input"
            />
            {query && (
              <button 
                type="button" 
                onClick={() => setQuery('')}
                style={{ position: 'absolute', right: '1rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: 'var(--text-muted)' }}
              >
                <X size={15} />
              </button>
            )}
          </div>

          {/* Configuration controls */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid var(--border-color)', paddingTop: '0.85rem' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', alignItems: 'center' }}>
              
              {/* Top-K configuration slider */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <Sliders size={13} style={{ color: 'var(--text-muted)' }} />
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', width: '50px' }}>Top K: {topK}</span>
                <input 
                  type="range" 
                  min="1" 
                  max="30" 
                  value={topK} 
                  onChange={e => setTopK(parseInt(e.target.value))}
                  style={{
                    accentColor: '#0f172a',
                    width: '80px',
                    height: '3px',
                    cursor: 'pointer'
                  }}
                />
              </div>

              {/* Reranker Toggle */}
              {stats?.reranker.enabled && (
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', userSelect: 'none' }}>
                  <input 
                    type="checkbox" 
                    checked={useRerank} 
                    onChange={e => setUseRerank(e.target.checked)}
                    style={{ accentColor: '#0f172a', width: '13px', height: '13px', cursor: 'pointer' }}
                  />
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Reranker
                  </span>
                </label>
              )}

              {/* View Layout Mode (Grouped vs Merged) */}
              <div style={{ display: 'flex', borderRadius: '6px', border: '1px solid var(--border-color)', overflow: 'hidden', padding: '1px', backgroundColor: 'var(--bg-tertiary)' }}>
                <button
                  type="button"
                  onClick={() => setDisplayMode('unified')}
                  style={{
                    border: 'none',
                    padding: '0.25rem 0.6rem',
                    borderRadius: '5px',
                    fontSize: '0.75rem',
                    fontWeight: 500,
                    backgroundColor: displayMode === 'unified' ? 'var(--bg-primary)' : 'transparent',
                    color: displayMode === 'unified' ? 'var(--text-primary)' : 'var(--text-secondary)',
                    boxShadow: displayMode === 'unified' ? '0 1px 2px rgba(0,0,0,0.05)' : 'none'
                  }}
                >
                  Unified
                </button>
                <button
                  type="button"
                  onClick={() => setDisplayMode('grouped')}
                  style={{
                    border: 'none',
                    padding: '0.25rem 0.6rem',
                    borderRadius: '5px',
                    fontSize: '0.75rem',
                    fontWeight: 500,
                    backgroundColor: displayMode === 'grouped' ? 'var(--bg-primary)' : 'transparent',
                    color: displayMode === 'grouped' ? 'var(--text-primary)' : 'var(--text-secondary)',
                    boxShadow: displayMode === 'grouped' ? '0 1px 2px rgba(0,0,0,0.05)' : 'none'
                  }}
                >
                  Grouped
                </button>
              </div>

            </div>

            <button 
              type="submit" 
              disabled={searchLoading || !query.trim()}
              className="btn-minimal-dark" 
              style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.5rem 1.25rem' }}
            >
              {searchLoading ? <span className="spinner"></span> : <Search size={14} />}
              Search
            </button>
          </div>
        </form>

        {/* SEARCH RESULTS VIEW */}
        {searchLoading && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '5rem 0', gap: '0.85rem' }}>
            <span className="spinner" style={{ width: '30px', height: '30px', color: 'var(--text-secondary)', borderWidth: '2px' }}></span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Retrieving document nodes...
            </span>
          </div>
        )}

        {searchError && (
          <div style={{
            padding: '1rem',
            backgroundColor: '#fef2f2',
            border: '1px solid #fee2e2',
            borderRadius: '8px',
            color: 'var(--accent-rose)',
            fontSize: '0.85rem'
          }}>
            {searchError}
          </div>
        )}

        {!searchLoading && !searchError && results && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            
            {/* Unified Mode Layout */}
            {displayMode === 'unified' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <h3 style={{ fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.4rem' }}>
                  <Layers size={15} style={{ color: 'var(--text-secondary)' }} />
                  Merged Ranking Results
                  <span style={{ fontSize: '0.725rem', fontWeight: 400, color: 'var(--text-muted)' }}>
                    (Showing top {results.hits.length} hits scored on a single scale)
                  </span>
                </h3>

                {results.hits.length === 0 ? (
                  <div className="flat-card" style={{ textAlign: 'center', padding: '2.5rem', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                    No results found for your query. Try syncing the database or adjusting terms.
                  </div>
                ) : (
                  results.hits.map((hit, index) => (
                    <div key={index} className="flat-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                      
                      {/* Hit Header Metas */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                          <span style={{ fontSize: '0.7rem', fontWeight: 700, backgroundColor: 'var(--bg-tertiary)', width: '18px', height: '18px', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            {index + 1}
                          </span>
                          
                          <span className={`badge ${hit.source === 'code' ? 'badge-code' : 'badge-prose'}`}>
                            {hit.source}
                          </span>

                          <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.8rem', fontWeight: 600 }}>
                            {getFileIcon(hit.doc_id)}
                            {hit.doc_id}
                          </span>

                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                            [{hit.locator}]
                          </span>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          {getScoreTag(hit)}
                          
                          <button 
                            onClick={() => handleCopyText(hit.text, `hit-${index}`)}
                            title="Copy chunk text"
                            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', padding: '0.2rem', cursor: 'pointer' }}
                          >
                            {copiedId === `hit-${index}` ? <Check size={12} style={{ color: 'var(--accent-emerald)' }} /> : <Copy size={12} />}
                          </button>

                          <button 
                            onClick={() => openFilePreview(hit.doc_id)}
                            className="btn-minimal-outline" 
                            style={{ padding: '0.25rem 0.5rem', fontSize: '0.725rem', display: 'flex', alignItems: 'center', gap: '0.25rem', borderRadius: '4px' }}
                          >
                            View File
                            <ExternalLink size={10} />
                          </button>
                        </div>
                      </div>

                      {/* Code Block Content */}
                      <div className="code-preview-block">
                        <div className="code-body" style={{ maxHeight: '160px', overflowY: 'auto' }}>
                          {hit.text}
                        </div>
                      </div>

                    </div>
                  ))
                )}
              </div>
            )}

            {/* Grouped Domain Mode Layout */}
            {displayMode === 'grouped' && results.grouped && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
                
                {/* PROSE DOMAIN HITS */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                  <h3 style={{ fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.4rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.4rem', color: 'var(--text-primary)' }}>
                    <FileText size={14} style={{ color: 'var(--text-secondary)' }} />
                    Prose Collection
                    <span style={{ fontSize: '0.7rem', fontWeight: 400, color: 'var(--text-muted)' }}>
                      ({results.grouped.prose.length} hits)
                    </span>
                  </h3>

                  {results.grouped.prose.length === 0 ? (
                    <div className="flat-card" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                      No prose records matched.
                    </div>
                  ) : (
                    results.grouped.prose.map((hit, idx) => (
                      <div key={`prose-${idx}`} className="flat-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.4rem' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap', maxWidth: '80%' }}>
                            {getFileIcon(hit.doc_id)}
                            <span style={{ fontSize: '0.75rem', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={hit.doc_id}>
                              {hit.doc_id.split('/').pop()}
                            </span>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                              [{hit.locator}]
                            </span>
                          </div>
                          <span className="badge badge-score" style={{ fontSize: '0.65rem' }}>
                            Cos: {hit.score.toFixed(3)}
                          </span>
                        </div>

                        <div className="code-preview-block">
                          <div className="code-body" style={{ maxHeight: '100px', fontSize: '0.75rem', padding: '0.5rem' }}>
                            {hit.text}
                          </div>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.4rem' }}>
                          <button 
                            onClick={() => openFilePreview(hit.doc_id)}
                            className="btn-minimal-outline" 
                            style={{ padding: '0.2rem 0.4rem', fontSize: '0.7rem', borderRadius: '4px' }}
                          >
                            Inspect
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>

                {/* CODE DOMAIN HITS */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                  <h3 style={{ fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.4rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.4rem', color: 'var(--text-primary)' }}>
                    <Code size={14} style={{ color: 'var(--text-secondary)' }} />
                    Code Collection
                    <span style={{ fontSize: '0.7rem', fontWeight: 400, color: 'var(--text-muted)' }}>
                      ({results.grouped.code.length} hits)
                    </span>
                  </h3>

                  {results.grouped.code.length === 0 ? (
                    <div className="flat-card" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                      No code files matched.
                    </div>
                  ) : (
                    results.grouped.code.map((hit, idx) => (
                      <div key={`code-${idx}`} className="flat-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.4rem' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap', maxWidth: '80%' }}>
                            {getFileIcon(hit.doc_id)}
                            <span style={{ fontSize: '0.75rem', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={hit.doc_id}>
                              {hit.doc_id.split('/').pop()}
                            </span>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                              [{hit.locator}]
                            </span>
                          </div>
                          <span className="badge badge-score" style={{ fontSize: '0.65rem' }}>
                            Cos: {hit.score.toFixed(3)}
                          </span>
                        </div>

                        <div className="code-preview-block">
                          <div className="code-body" style={{ maxHeight: '100px', fontSize: '0.75rem', padding: '0.5rem' }}>
                            {hit.text}
                          </div>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.4rem' }}>
                          <button 
                            onClick={() => openFilePreview(hit.doc_id)}
                            className="btn-minimal-outline" 
                            style={{ padding: '0.2rem 0.4rem', fontSize: '0.7rem', borderRadius: '4px' }}
                          >
                            Inspect
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>

              </div>
            )}

          </div>
        )}

      </main>

      {/* FULL SOURCE CODE VIEWER DRAWER / MODAL */}
      {previewDocId && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(15, 23, 42, 0.4)',
          display: 'flex',
          justifyContent: 'flex-end',
          zIndex: 1000
        }}>
          <div style={{
            width: '60%',
            maxWidth: '900px',
            minWidth: '400px',
            height: '100%',
            backgroundColor: 'var(--bg-primary)',
            borderLeft: '1px solid var(--border-color)',
            boxShadow: '-4px 0 24px rgba(0, 0, 0, 0.08)',
            display: 'flex',
            flexDirection: 'column',
            padding: '1.5rem'
          }}>
            {/* Header controls */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <FileCode size={18} style={{ color: 'var(--text-secondary)' }} />
                <div>
                  <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>{previewDocId.split('/').pop()}</h3>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{previewDocId}</span>
                </div>
              </div>
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {previewContent && (
                  <button 
                    onClick={() => handleCopyText(previewContent, 'preview-doc')}
                    className="btn-minimal-outline" 
                    style={{ padding: '0.35rem 0.75rem', fontSize: '0.725rem', display: 'flex', alignItems: 'center', gap: '0.3rem', borderRadius: '5px' }}
                  >
                    {copiedId === 'preview-doc' ? <Check size={12} style={{ color: 'var(--accent-emerald)' }} /> : <Copy size={12} />}
                    {copiedId === 'preview-doc' ? 'Copied' : 'Copy File'}
                  </button>
                )}
                
                <button 
                  onClick={() => setPreviewDocId(null)}
                  style={{ background: 'var(--bg-tertiary)', border: 'none', color: 'var(--text-primary)', padding: '0.35rem', borderRadius: '6px', cursor: 'pointer' }}
                >
                  <X size={15} />
                </button>
              </div>
            </div>

            {/* Content box */}
            <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
              {previewLoading ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: '0.85rem' }}>
                  <span className="spinner" style={{ color: 'var(--text-muted)' }}></span>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Loading document...</span>
                </div>
              ) : previewError ? (
                <div style={{ padding: '1rem', backgroundColor: '#fef2f2', border: '1px solid #fee2e2', borderRadius: '6px', color: 'var(--accent-rose)', fontSize: '0.85rem' }}>
                  {previewError}
                </div>
              ) : previewContent ? (
                <div className="code-preview-block" style={{ flex: 1, margin: 0, display: 'flex', flexDirection: 'column' }}>
                  <div className="code-header" style={{ padding: '0.3rem 0.75rem' }}>
                    <span>Source Content Stream</span>
                    <span style={{ fontSize: '0.65rem' }}>Max (500KB)</span>
                  </div>
                  <div className="code-body" style={{ flex: 1, overflowY: 'auto', padding: '0.75rem', margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                    {previewContent}
                  </div>
                </div>
              ) : null}
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
