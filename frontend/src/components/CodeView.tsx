import { useMemo, useState } from "react";

const TOKEN = /(#.*$)|('[^']*'|"[^"]*")|(\b-?\d+(?:\.\d+)?(?:e[-+]?\d+)?\b)|(\b(?:from|import|for|in|range|None|True|False)\b)|([A-Za-z_][\w.]*(?=\())|([A-Za-z_]\w*(?==))|([()[\],=.:+*/-])/g;

function highlight(line: string) {
  const out: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  TOKEN.lastIndex = 0;
  while ((m = TOKEN.exec(line))) {
    if (m.index > last) out.push(line.slice(last, m.index));
    const cls = m[1] ? "tok-c" : m[2] ? "tok-s" : m[3] ? "tok-n" : m[4] ? "tok-k" : m[5] ? "tok-f" : m[6] ? "tok-a" : "tok-p";
    out.push(<span key={m.index} className={cls}>{m[0]}</span>);
    last = m.index + m[0].length;
    if (m[0].length === 0) TOKEN.lastIndex++;
  }
  if (last < line.length) out.push(line.slice(last));
  return out;
}

/** Syntax-highlighted Python. `compare` marks lines that differ from another version of the same source. */
export default function CodeView({ code, compare, exact, filename, maxHeight }: {
  code: string;
  compare?: string | null;
  exact?: string;
  filename: string;
  maxHeight?: number;
}) {
  const lines = useMemo(() => code.replace(/\n$/, "").split("\n"), [code]);
  const other = useMemo(() => (compare ? compare.replace(/\n$/, "").split("\n") : null), [compare]);
  const [copied, setCopied] = useState(false);
  const full = exact ?? code;
  const download = () => {
    const url = URL.createObjectURL(new Blob([full], { type: "text/x-python" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="window" style={{ display: "flex", flexDirection: "column" }}>
      <div className="bar">
        <i /> <i /> <i />
        <span className="mono" style={{ marginLeft: 8, color: "var(--ink-2)" }}>{filename}</span>
        <span className="spacer" />
        <button className="btn sm ghost" onClick={async () => { await navigator.clipboard?.writeText(full); setCopied(true); setTimeout(() => setCopied(false), 1400); }}>
          {copied ? "Copied" : "Copy"}
        </button>
        <button className="btn sm ghost" onClick={download}>Download .py</button>
      </div>
      <pre className="code" style={{ maxHeight }} aria-label={`${filename} source`}>
        {lines.map((l, i) => {
          const chg = other && other[i] !== undefined && other[i] !== l && !l.startsWith("#");
          return <span key={i} className={`ln ${chg ? "chg" : ""}`}>{highlight(l)}{"\n"}</span>;
        })}
      </pre>
    </div>
  );
}
