/**
 * renderer.ts — mcp-lubricant
 *
 * 目的: LLMが50個以上のMCPツールからコンテキスト崩壊せずに
 *       正しいツールを選べるよう、ツール群をSVGグラフに変換する。
 *
 * 完全にverantyx非依存。どのMCPサーバーにも追加できる。
 */

// ─────────────────────────────────────────────────────────────────────────────
// 入力型 (どんなMCPのツールリストでも受け付ける)
// ─────────────────────────────────────────────────────────────────────────────

export interface LubricantTool {
    name: string;
    description: string;
    category?: string;      // グループ名。未指定なら description から自動推定
    usedAfter?: string[];   // このツールの前に使われることが多いツール名
    usedBefore?: string[];  // このツールの後に使われることが多いツール名
    relevanceScore?: number; // クエリとの関連スコア (0.0〜1.0)
}

export interface RenderOptions {
    query?: string;          // LLMが今やりたいこと (ハイライトに使う)
    currentTool?: string;    // 今実行中のツール名 (赤でマーク)
    maxVisible?: number;     // 表示最大ツール数 (多すぎる場合に絞る, default: 20)
    showEdges?: boolean;     // ツール間のエッジを表示するか (default: true)
}

// ─────────────────────────────────────────────────────────────────────────────
// カテゴリ推定 (description のキーワードから)
// ─────────────────────────────────────────────────────────────────────────────

const CATEGORY_KEYWORDS: Record<string, string[]> = {
    "search":  ["search", "find", "query", "lookup", "fetch", "get", "read", "list"],
    "write":   ["write", "save", "store", "create", "insert", "add", "update", "edit", "delete", "remove"],
    "run":     ["run", "exec", "execute", "start", "launch", "call", "invoke", "trigger", "send"],
    "analyze": ["analyze", "check", "lint", "test", "validate", "review", "inspect", "debug", "diff"],
    "meta":    ["config", "setup", "init", "install", "manage", "status", "info", "help", "list tools"],
};

export function inferCategory(tool: LubricantTool): string {
    if (tool.category) return tool.category;
    const desc = (tool.name + " " + tool.description).toLowerCase();
    for (const [cat, kws] of Object.entries(CATEGORY_KEYWORDS)) {
        if (kws.some(k => desc.includes(k))) return cat;
    }
    return "other";
}

// ─────────────────────────────────────────────────────────────────────────────
// クエリ関連スコアリング (キーワードマッチ)
// ─────────────────────────────────────────────────────────────────────────────

export function scoreToolForQuery(tool: LubricantTool, query: string): number {
    if (!query) return tool.relevanceScore ?? 0.3;
    const q = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    const text = (tool.name + " " + tool.description).toLowerCase();
    const hits = q.filter(w => text.includes(w)).length;
    return Math.min(hits / Math.max(q.length, 1), 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// SVG レンダラー
// ─────────────────────────────────────────────────────────────────────────────

const W = 960;
const HEADER_H = 48;
const FOOTER_H = 32;
const CARD_W = 170;
const CARD_H = 44;
const COL_PAD = 16;
const ROW_PAD = 12;

const CAT_COLORS: Record<string, { bg: string; border: string; text: string; label: string }> = {
    search:  { bg: "#0f172a", border: "#3b82f6", text: "#93c5fd", label: "SEARCH / READ" },
    write:   { bg: "#0f1a0f", border: "#22c55e", text: "#86efac", label: "WRITE / SAVE" },
    run:     { bg: "#1a0f0f", border: "#f97316", text: "#fdba74", label: "RUN / EXEC" },
    analyze: { bg: "#170f1a", border: "#a855f7", text: "#d8b4fe", label: "ANALYZE / CHECK" },
    meta:    { bg: "#0f0f1a", border: "#64748b", text: "#94a3b8", label: "META / CONFIG" },
    other:   { bg: "#111111", border: "#475569", text: "#64748b", label: "OTHER" },
};

function escXml(s: string) {
    return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function truncate(s: string, n: number) {
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

export function renderToolMap(tools: LubricantTool[], opts: RenderOptions = {}): string {
    const { query = "", currentTool, maxVisible = 20, showEdges = true } = opts;

    // スコアリング & 絞り込み
    const scored = tools
        .map(t => ({ ...t, _score: scoreToolForQuery(t, query), _cat: inferCategory(t) }))
        .sort((a, b) => b._score - a._score)
        .slice(0, maxVisible);

    // カテゴリ別にグループ化
    const groups: Record<string, typeof scored> = {};
    for (const t of scored) {
        if (!groups[t._cat]) groups[t._cat] = [];
        groups[t._cat].push(t);
    }

    const catOrder = ["search", "write", "run", "analyze", "meta", "other"];
    const activeCats = catOrder.filter(c => groups[c]?.length);

    // 列レイアウト計算
    const colW = Math.floor((W - COL_PAD * (activeCats.length + 1)) / activeCats.length);
    const maxRows = Math.max(...activeCats.map(c => groups[c]?.length ?? 0));
    const totalH = HEADER_H + 28 + maxRows * (CARD_H + ROW_PAD) + ROW_PAD + FOOTER_H;

    // 座標マップ
    const coords = new Map<string, { x: number; y: number; cat: string }>();
    activeCats.forEach((cat, ci) => {
        const x = COL_PAD + ci * (colW + COL_PAD);
        (groups[cat] ?? []).forEach((t, ri) => {
            coords.set(t.name, {
                x,
                y: HEADER_H + 28 + ri * (CARD_H + ROW_PAD),
                cat,
            });
        });
    });

    let svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${totalH}" viewBox="0 0 ${W} ${totalH}">
<rect width="${W}" height="${totalH}" fill="#060a14"/>

<!-- Header -->
<rect width="${W}" height="${HEADER_H}" fill="#0a1020"/>
<rect y="${HEADER_H - 1}" width="${W}" height="1" fill="#1e3a5f" opacity="0.8"/>
<text x="16" y="20" font-family="monospace" font-size="13" fill="#3b82f6" font-weight="bold">⬡ MCP LUBRICANT</text>
<text x="16" y="36" font-family="monospace" font-size="9" fill="#334155">TOOL ROUTING MAP · ${scored.length}/${tools.length} tools shown</text>
${query ? `<text x="${W - 16}" y="20" text-anchor="end" font-family="monospace" font-size="9" fill="#60a5fa">query: "${escXml(truncate(query, 50))}"</text>` : ""}
${currentTool ? `<text x="${W - 16}" y="36" text-anchor="end" font-family="monospace" font-size="9" fill="#ef4444">▶ ${escXml(currentTool)}</text>` : ""}

<!-- Column headers -->
${activeCats.map((cat, ci) => {
    const x = COL_PAD + ci * (colW + COL_PAD);
    const c = CAT_COLORS[cat];
    return `<rect x="${x}" y="${HEADER_H + 4}" width="${colW}" height="18" rx="3" fill="${c.border}" opacity="0.12"/>
<text x="${x + colW/2}" y="${HEADER_H + 16}" text-anchor="middle"
      font-family="monospace" font-size="8" fill="${c.border}" letter-spacing="1">${c.label}</text>`;
}).join("\n")}

`;

    // エッジ描画
    if (showEdges) {
        svg += "<!-- Edges -->\n<defs><marker id='arr' markerWidth='5' markerHeight='5' refX='4' refY='2.5' orient='auto'><path d='M0,0 L5,2.5 L0,5 Z' fill='#1e3a5f'/></marker></defs>\n";
        for (const t of scored) {
            const from = coords.get(t.name);
            if (!from) continue;
            for (const next of (t.usedBefore ?? [])) {
                const to = coords.get(next);
                if (!to) continue;
                const fx = from.x + CARD_W;
                const fy = from.y + CARD_H / 2;
                const tx = to.x;
                const ty = to.y + CARD_H / 2;
                const mx = (fx + tx) / 2;
                svg += `<path d="M${fx},${fy} C${mx},${fy} ${mx},${ty} ${tx},${ty}"
                       fill="none" stroke="#1e3a5f" stroke-width="1.2"
                       marker-end="url(#arr)" opacity="0.5"/>\n`;
            }
        }
    }

    // ノード描画
    svg += "\n<!-- Tool nodes -->\n";
    for (const t of scored) {
        const c = coords.get(t.name);
        if (!c) continue;

        const col = CAT_COLORS[c.cat];
        const isCurrent   = t.name === currentTool;
        const isHighlited = t._score > 0.4 && !isCurrent;

        const borderColor = isCurrent   ? "#ef4444"
                          : isHighlited ? "#facc15"
                          : col.border;
        const bgColor     = isCurrent   ? "#3f0000"
                          : isHighlited ? "#2a2000"
                          : col.bg;
        const strokeW     = isCurrent ? 2.5 : isHighlited ? 1.5 : 1;

        // スコアバー (右端)
        const barW = Math.round((CARD_W - 20) * t._score);

        svg += `
<!-- ${t.name} -->
${isCurrent ? `<rect x="${c.x-3}" y="${c.y-3}" width="${CARD_W+6}" height="${CARD_H+6}" rx="7" fill="none" stroke="#ef4444" stroke-width="1" opacity="0.4"/>` : ""}
<rect x="${c.x}" y="${c.y}" width="${CARD_W}" height="${CARD_H}" rx="5"
      fill="${bgColor}" stroke="${borderColor}" stroke-width="${strokeW}"/>
<text x="${c.x+8}" y="${c.y+16}" font-family="monospace" font-size="10" font-weight="bold" fill="${isCurrent ? "#fca5a5" : isHighlited ? "#fef08a" : col.text}">${escXml(truncate(t.name, 18))}</text>
<text x="${c.x+8}" y="${c.y+30}" font-family="monospace" font-size="7" fill="${col.text}" opacity="0.55">${escXml(truncate(t.description, 26))}</text>
${query && t._score > 0 ? `<rect x="${c.x+8}" y="${c.y+36}" width="${barW}" height="3" rx="1" fill="${borderColor}" opacity="0.5"/>` : ""}
`;
    }

    // フッター
    const fy = totalH - FOOTER_H;
    svg += `
<!-- Footer -->
<rect y="${fy}" width="${W}" height="${FOOTER_H}" fill="#0a1020"/>
<rect y="${fy}" width="${W}" height="1" fill="#1e3a5f" opacity="0.5"/>
<text x="16" y="${fy + 13}" font-family="monospace" font-size="8" fill="#ef4444">■ RED=current  </text>
<text x="108" y="${fy + 13}" font-family="monospace" font-size="8" fill="#facc15">■ YELLOW=best match  </text>
<text x="260" y="${fy + 13}" font-family="monospace" font-size="8" fill="#475569">■ DIM=less relevant</text>
<text x="16" y="${fy + 25}" font-family="monospace" font-size="8" fill="#1e3a5f">INSTRUCTION: Pick the YELLOW tool node ID that best matches your query. Output the tool name only.</text>
</svg>`;

    return svg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Base64 export (sharp optional, fallback to SVG)
// ─────────────────────────────────────────────────────────────────────────────

export async function toBase64(svg: string): Promise<{ base64: string; mimeType: string }> {
    try {
        const sharp = (await import("sharp")).default;
        const buf = await sharp(Buffer.from(svg)).png().toBuffer();
        return { base64: buf.toString("base64"), mimeType: "image/png" };
    } catch {
        return { base64: Buffer.from(svg).toString("base64"), mimeType: "image/svg+xml" };
    }
}
