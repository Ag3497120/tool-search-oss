/**
 * search.ts — mcp-lubricant
 *
 * BM25ベースのツール検索エンジン。
 *
 * 核心アイデア (defer_loading パターン):
 *   - LLMは最初に全ツール定義を受け取らない
 *   - search_tools(query) で上位N件だけ取得する
 *   - コンテキスト使用量: 50ツール → 3〜5ツール = 最大85%削減
 *
 * 依存ゼロ。pure TypeScript実装。
 */

export interface ToolDefinition {
    name: string;
    description: string;
    inputSchema?: Record<string, unknown>;
    category?: string;
    tags?: string[];
}

export interface SearchResult {
    tool: ToolDefinition;
    score: number;
    matchedTerms: string[];
}

// ─────────────────────────────────────────────────────────────────────────────
// BM25 実装 (k1=1.5, b=0.75)
// ─────────────────────────────────────────────────────────────────────────────

const K1 = 1.5;
const B  = 0.75;

function tokenize(text: string): string[] {
    return text
        .toLowerCase()
        .replace(/[_\-\/\\]/g, " ")   // snake_case / kebab-case を分割
        .replace(/([a-z])([A-Z])/g, "$1 $2")  // camelCase を分割
        .replace(/[^a-z0-9 ]/g, " ")
        .split(/\s+/)
        .filter(t => t.length > 1);
}

function buildCorpus(tools: ToolDefinition[]): {
    tf: Map<string, Map<string, number>>;
    df: Map<string, number>;
    avgDl: number;
    docLengths: Map<string, number>;
} {
    const tf  = new Map<string, Map<string, number>>();  // doc → term → freq
    const df  = new Map<string, number>();                // term → doc count
    const docLengths = new Map<string, number>();

    for (const tool of tools) {
        const text = [
            tool.name,
            tool.description,
            tool.category ?? "",
            (tool.tags ?? []).join(" "),
        ].join(" ");
        const tokens = tokenize(text);
        docLengths.set(tool.name, tokens.length);

        const termFreq = new Map<string, number>();
        for (const t of tokens) termFreq.set(t, (termFreq.get(t) ?? 0) + 1);
        tf.set(tool.name, termFreq);

        for (const term of termFreq.keys()) {
            df.set(term, (df.get(term) ?? 0) + 1);
        }
    }

    const avgDl = [...docLengths.values()].reduce((a, b) => a + b, 0) / (docLengths.size || 1);
    return { tf, df, avgDl, docLengths };
}

function bm25Score(
    query: string[],
    docName: string,
    corpus: ReturnType<typeof buildCorpus>,
    N: number
): { score: number; matchedTerms: string[] } {
    const { tf, df, avgDl, docLengths } = corpus;
    const docTf = tf.get(docName) ?? new Map();
    const dl    = docLengths.get(docName) ?? 0;
    const matched: string[] = [];
    let score = 0;

    for (const term of query) {
        const freq = docTf.get(term) ?? 0;
        if (freq === 0) continue;
        matched.push(term);

        const idf = Math.log((N - (df.get(term) ?? 0) + 0.5) / ((df.get(term) ?? 0) + 0.5) + 1);
        const tf_norm = (freq * (K1 + 1)) / (freq + K1 * (1 - B + B * (dl / avgDl)));
        score += idf * tf_norm;
    }

    return { score, matchedTerms: matched };
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolSearchEngine — メインクラス
// ─────────────────────────────────────────────────────────────────────────────

export class ToolSearchEngine {
    private tools: ToolDefinition[];
    private corpus: ReturnType<typeof buildCorpus>;

    constructor(tools: ToolDefinition[]) {
        this.tools  = tools;
        this.corpus = buildCorpus(tools);
    }

    /**
     * クエリに最も合うツールをBM25スコアで返す。
     * topK: 返す件数 (default: 5)
     * threshold: スコアがこれ未満のツールを除外 (default: 0)
     */
    search(query: string, topK = 5, threshold = 0): SearchResult[] {
        const queryTerms = tokenize(query);
        const N = this.tools.length;

        const results: SearchResult[] = [];
        for (const tool of this.tools) {
            const { score, matchedTerms } = bm25Score(queryTerms, tool.name, this.corpus, N);
            if (score > threshold) results.push({ tool, score, matchedTerms });
        }

        return results
            .sort((a, b) => b.score - a.score)
            .slice(0, topK);
    }

    /**
     * 正規表現による高速フィルタ (BM25の前段階として使う)
     */
    regexFilter(pattern: string): ToolDefinition[] {
        try {
            const re = new RegExp(pattern, "i");
            return this.tools.filter(t =>
                re.test(t.name) || re.test(t.description) || re.test(t.category ?? "")
            );
        } catch {
            // invalid regex → キーワードフィルタにフォールバック
            return this.tools.filter(t =>
                t.name.toLowerCase().includes(pattern.toLowerCase()) ||
                t.description.toLowerCase().includes(pattern.toLowerCase())
            );
        }
    }

    /**
     * defer_loading パターン:
     * 全ツールの「軽量サマリー」だけを返す (name + 1行説明のみ)
     * これをLLMのコンテキストに入れ、次に search() で絞り込む。
     */
    summary(maxChars = 3000): string {
        const lines = this.tools.map(t =>
            `${t.name}: ${t.description.slice(0, 60)}`
        );
        let out = "";
        for (const line of lines) {
            if ((out + line).length > maxChars) { out += `\n... (${this.tools.length} tools total, use search_tools to filter)`; break; }
            out += line + "\n";
        }
        return out;
    }

    /** ツール数 */
    get size() { return this.tools.length; }
}
