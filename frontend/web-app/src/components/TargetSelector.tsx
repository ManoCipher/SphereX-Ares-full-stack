import React from 'react';
import type { Target } from '../types';

interface TargetSelectorProps {
  targets: Target[];
  selectedTargetId: string;
  onSelectTarget: (t: Target) => void;
  onClearTarget: () => void;
  ra: string;
  dec: string;
  radius: string;
  maxResults: string;
  startDate: string;
  endDate: string;
  onChangeRa: (v: string) => void;
  onChangeDec: (v: string) => void;
  onChangeRadius: (v: string) => void;
  onChangeMaxResults: (v: string) => void;
  onChangeStartDate: (v: string) => void;
  onChangeEndDate: (v: string) => void;
  onQuery: () => void;
  isQuerying: boolean;
  onSmartSelect: () => void;
  isSmartSelecting: boolean;
  smartExplanation?: string;
  statusMessage?: string;
}

export const TargetSelector: React.FC<TargetSelectorProps> = ({
  targets,
  selectedTargetId,
  onSelectTarget,
  onClearTarget,
  ra,
  dec,
  radius,
  maxResults,
  startDate,
  endDate,
  onChangeRa,
  onChangeDec,
  onChangeRadius,
  onChangeMaxResults,
  onChangeStartDate,
  onChangeEndDate,
  onQuery,
  isQuerying,
  onSmartSelect,
  isSmartSelecting,
  smartExplanation,
  statusMessage,
}) => {
  const hasCoordinates = Boolean(ra.trim() && dec.trim());
  const activeTarget = targets.find((t) => t.id === selectedTargetId);

  return (
    <div style={{
      padding: '14px 16px',
      borderRadius: '8px',
      border: '1px solid #1e293b',
      backgroundColor: '#0c1322',
      display: 'flex',
      flexDirection: 'column',
      gap: '12px',
      fontSize: '12px',
    }}>
      {/* Benchmark Target Quick-Pills */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: '#94a3b8', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 'bold' }}>
            Preset Objects:
          </span>
          <span style={{ color: '#64748b', fontSize: '10px' }}>(Click to autofill coordinates)</span>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center' }}>
          {targets.map((t) => {
            const active = t.id === selectedTargetId;
            return (
              <button
                key={t.id}
                onClick={() => onSelectTarget(t)}
                title={`Target: ${t.name}\nRA: ${t.ra_deg.toFixed(4)}°, Dec: ${t.dec_deg.toFixed(4)}°\n${t.source_note || ''}`}
                style={{
                  padding: '4px 10px',
                  borderRadius: '4px',
                  border: active ? '1px solid #38bdf8' : '1px solid #1e293b',
                  backgroundColor: active ? 'rgba(56, 189, 248, 0.18)' : '#070b14',
                  color: active ? '#38bdf8' : '#cbd5e1',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontFamily: 'monospace',
                  fontWeight: active ? 'bold' : 'normal',
                  transition: 'all 0.15s ease',
                }}
              >
                {t.name}
              </button>
            );
          })}

          {selectedTargetId && (
            <button
              onClick={onClearTarget}
              title="Clear active preset and enter custom coordinates"
              style={{
                padding: '4px 8px',
                borderRadius: '4px',
                border: '1px solid #334155',
                backgroundColor: 'transparent',
                color: '#94a3b8',
                cursor: 'pointer',
                fontSize: '10px',
                fontFamily: 'monospace',
              }}
            >
              Clear Preset
            </button>
          )}
        </div>
      </div>

      {/* Target Description Banner when active */}
      {activeTarget && (
        <div style={{
          padding: '6px 10px',
          backgroundColor: 'rgba(56, 189, 248, 0.08)',
          borderLeft: '3px solid #38bdf8',
          borderRadius: '2px',
          color: '#93c5fd',
          fontSize: '11px',
          fontFamily: 'monospace',
          display: 'flex',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px',
        }}>
          <span><strong>Active Target:</strong> {activeTarget.name}</span>
          <span style={{ color: '#cbd5e1' }}>
            PM: {activeTarget.pm_ra_cosdec_mas_yr ?? 0} mas/yr (RA), {activeTarget.pm_dec_mas_yr ?? 0} mas/yr (Dec)
          </span>
        </div>
      )}

      {/* Coordinate & Search Form */}
      <div className="ares-search-grid">
        <div>
          <label style={{ display: 'block', color: '#94a3b8', fontSize: '10px', marginBottom: '3px' }}>RA (deg)</label>
          <input
            type="text"
            value={ra}
            title="Right Ascension in decimal degrees (e.g. 133.785)"
            onChange={(e) => onChangeRa(e.target.value)}
            placeholder="e.g. 133.785"
            style={{ width: '100%', boxSizing: 'border-box', padding: '6px 8px', backgroundColor: '#070b14', border: '1px solid #1e293b', color: '#fff', borderRadius: '4px', fontFamily: 'monospace' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', color: '#94a3b8', fontSize: '10px', marginBottom: '3px' }}>Dec (deg)</label>
          <input
            type="text"
            value={dec}
            title="Declination in decimal degrees (e.g. -7.244)"
            onChange={(e) => onChangeDec(e.target.value)}
            placeholder="e.g. -7.244"
            style={{ width: '100%', boxSizing: 'border-box', padding: '6px 8px', backgroundColor: '#070b14', border: '1px solid #1e293b', color: '#fff', borderRadius: '4px', fontFamily: 'monospace' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', color: '#94a3b8', fontSize: '10px', marginBottom: '3px' }}>Radius / FOV (°)</label>
          <input
            type="text"
            value={radius}
            title="Search radius / cutout box in decimal degrees (0.05° to 0.08°)"
            onChange={(e) => onChangeRadius(e.target.value)}
            placeholder="e.g. 0.08"
            style={{ width: '100%', boxSizing: 'border-box', padding: '6px 8px', backgroundColor: '#070b14', border: '1px solid #1e293b', color: '#fff', borderRadius: '4px', fontFamily: 'monospace' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', color: '#94a3b8', fontSize: '10px', marginBottom: '3px' }}>Exposures</label>
          <input
            type="number"
            min="2"
            max="12"
            value={maxResults}
            title="Maximum number of observation cutouts to fetch (2 to 12)"
            onChange={(e) => onChangeMaxResults(e.target.value)}
            placeholder="e.g. 6"
            style={{ width: '100%', boxSizing: 'border-box', padding: '6px 8px', backgroundColor: '#070b14', border: '1px solid #1e293b', color: '#fff', borderRadius: '4px', fontFamily: 'monospace' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', color: '#94a3b8', fontSize: '10px', marginBottom: '3px' }}>Start Date (opt)</label>
          <input
            type="text"
            value={startDate}
            title="Optional ISO observation start date filter (YYYY-MM-DD)"
            onChange={(e) => onChangeStartDate(e.target.value)}
            placeholder="YYYY-MM-DD"
            style={{ width: '100%', boxSizing: 'border-box', padding: '6px 8px', backgroundColor: '#070b14', border: '1px solid #1e293b', color: '#fff', borderRadius: '4px', fontFamily: 'monospace' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', color: '#94a3b8', fontSize: '10px', marginBottom: '3px' }}>End Date (opt)</label>
          <input
            type="text"
            value={endDate}
            title="Optional ISO observation end date filter (YYYY-MM-DD)"
            onChange={(e) => onChangeEndDate(e.target.value)}
            placeholder="YYYY-MM-DD"
            style={{ width: '100%', boxSizing: 'border-box', padding: '6px 8px', backgroundColor: '#070b14', border: '1px solid #1e293b', color: '#fff', borderRadius: '4px', fontFamily: 'monospace' }}
          />
        </div>

        <div className="ares-actions-group" style={{ display: 'flex', gap: '6px' }}>
          <button
            onClick={onQuery}
            disabled={!hasCoordinates || isQuerying || isSmartSelecting}
            title={hasCoordinates ? "Query NASA IRSA SIA2 and fetch FITS cutouts for these coordinates" : "Select a preset or enter coordinates first"}
            style={{
              flex: 1,
              padding: '7px 10px',
              backgroundColor: hasCoordinates ? '#0284c7' : '#1e293b',
              color: hasCoordinates ? '#fff' : '#64748b',
              fontWeight: 'bold',
              borderRadius: '4px',
              border: 'none',
              cursor: (!hasCoordinates || isQuerying || isSmartSelecting) ? 'not-allowed' : 'pointer',
              opacity: (isQuerying || isSmartSelecting) ? 0.6 : 1,
            }}
          >
            {isQuerying ? 'Fetching...' : 'Fetch FITS'}
          </button>

          <button
            onClick={onSmartSelect}
            disabled={!hasCoordinates || isQuerying || isSmartSelecting}
            title={hasCoordinates ? "Auto-select optimal multi-epoch observations using backend selection heuristic" : "Select a preset or enter coordinates first"}
            style={{
              flex: 1,
              padding: '7px 10px',
              backgroundColor: hasCoordinates ? '#0d9488' : '#1e293b',
              color: hasCoordinates ? '#fff' : '#64748b',
              fontWeight: 'bold',
              borderRadius: '4px',
              border: 'none',
              cursor: (!hasCoordinates || isQuerying || isSmartSelecting) ? 'not-allowed' : 'pointer',
              opacity: (isQuerying || isSmartSelecting) ? 0.6 : 1,
            }}
          >
            {isSmartSelecting ? 'Selecting...' : 'Auto Select'}
          </button>
        </div>
      </div>

      {/* Rationale or Status Banner */}
      {smartExplanation && (
        <div style={{
          padding: '8px 12px',
          backgroundColor: '#0f172a',
          border: '1px solid #0d9488',
          borderRadius: '4px',
          color: '#5eead4',
          fontSize: '11px',
          fontFamily: 'monospace',
        }}>
          <strong>Selection Rationale:</strong> {smartExplanation}
        </div>
      )}

      {statusMessage && (
        <div style={{
          padding: '8px 12px',
          backgroundColor: '#0c1322',
          border: '1px solid #334155',
          borderRadius: '4px',
          color: '#cbd5e1',
          fontSize: '11px',
          fontFamily: 'monospace',
        }}>
          {statusMessage}
        </div>
      )}
    </div>
  );
};