import React, { useRef, useEffect } from 'react';

interface BlinkCanvasProps {
  imageSrc: string;
  activeFilename?: string;
  investigationMarker: { x: number; y: number } | null;
  onCanvasClick: (normX: number, normY: number) => void;
  isLoading?: boolean;
  loadingMessage?: string;
  isEmpty?: boolean;
}

export const BlinkCanvas: React.FC<BlinkCanvasProps> = ({
  imageSrc,
  investigationMarker,
  onCanvasClick,
  isLoading = false,
  loadingMessage = 'Loading...',
  isEmpty = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isLoading || isEmpty) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const normX = Math.max(0, Math.min(1, clickX / rect.width));
    const normY = Math.max(0, Math.min(1, clickY / rect.height));
    onCanvasClick(normX, normY);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (isEmpty || !imageSrc) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#050811';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageSrc;
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;

      ctx.fillStyle = '#050811';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw clean FITS image pixels
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Only draw marker if the user has explicitly clicked to investigate
      if (investigationMarker) {
        const mx = investigationMarker.x * canvas.width;
        const my = investigationMarker.y * canvas.height;

        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(mx, my, 12, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(mx - 16, my); ctx.lineTo(mx + 16, my);
        ctx.moveTo(mx, my - 16); ctx.lineTo(mx, my + 16);
        ctx.stroke();

        ctx.fillStyle = '#38bdf8';
        ctx.font = '10px monospace';
        ctx.fillText('CANDIDATE', mx + 14, my - 6);
      }
    };
  }, [imageSrc, investigationMarker, isEmpty]);

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      maxWidth: '100%',
      aspectRatio: '1/1',
      maxHeight: 'min(70vh, 520px)',
      backgroundColor: '#000',
      borderRadius: '8px',
      border: '1px solid #1e293b',
      overflow: 'hidden',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)',
      cursor: (isEmpty || isLoading) ? 'default' : 'crosshair',
    }}>
      <canvas
        ref={canvasRef}
        width={480}
        height={480}
        onClick={handleClick}
        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
      />

      {/* Loading Overlay */}
      {isLoading && (
        <div style={{
          position: 'absolute',
          inset: 0,
          backgroundColor: 'rgba(5, 8, 17, 0.88)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px',
          zIndex: 10,
          color: '#38bdf8',
          fontFamily: 'monospace',
          fontSize: '12px',
        }}>
          <div style={{
            width: '32px',
            height: '32px',
            border: '3px solid #1e293b',
            borderTop: '3px solid #38bdf8',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
          }} />
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
          <span>{loadingMessage}</span>
        </div>
      )}

      {/* Empty State Prompt */}
      {isEmpty && !isLoading && (
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '10px',
          padding: '24px',
          textAlign: 'center',
          color: '#64748b',
          fontFamily: 'monospace',
          fontSize: '12px',
          pointerEvents: 'none',
        }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '50%', border: '1px dashed #334155', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#38bdf8' }}>
            +
          </div>
          <strong style={{ color: '#cbd5e1' }}>No Observations Loaded</strong>
          <p style={{ maxWidth: '300px', margin: 0, lineHeight: '1.4' }}>
            Select a preset benchmark object above (e.g., WISE 0855 or Barnard's Star) or enter coordinates, then click <strong>Fetch FITS</strong> or <strong>Auto Select</strong>.
          </p>
        </div>
      )}

      {/* Click Guidance Prompt */}
      {!isEmpty && !isLoading && !investigationMarker && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'rgba(5, 8, 17, 0.75)',
          padding: '4px 12px',
          borderRadius: '9999px',
          fontSize: '10px',
          fontFamily: 'monospace',
          color: '#64748b',
          pointerEvents: 'none',
        }}>
          Click any source on the image to measure astrometry
        </div>
      )}
    </div>
  );
};