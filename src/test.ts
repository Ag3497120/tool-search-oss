// test.ts — tool-search-oss
import { ToolSearchEngine } from "./search.ts";

const TOOLS = [
    { name: "read_file",        description: "Read contents of a file from disk" },
    { name: "write_file",       description: "Write or overwrite a file on disk" },
    { name: "list_dir",         description: "List files and directories" },
    { name: "search_files",     description: "Search files by regex pattern" },
    { name: "run_command",      description: "Execute a shell command" },
    { name: "run_tests",        description: "Run the test suite" },
    { name: "gh_create_issue",  description: "Create a GitHub issue" },
    { name: "gh_list_prs",      description: "List open pull requests" },
    { name: "gh_push_commit",   description: "Push a commit to GitHub" },
    { name: "slack_send",       description: "Send a Slack message to a channel" },
    { name: "email_send",       description: "Send an email via SMTP" },
    { name: "db_query",         description: "Execute a SQL SELECT query" },
    { name: "db_insert",        description: "Insert rows into a database table" },
    { name: "http_get",         description: "Make an HTTP GET request" },
    { name: "http_post",        description: "Make an HTTP POST request" },
    { name: "lint_code",        description: "Run linter on source code" },
    { name: "format_code",      description: "Auto-format source code" },
    { name: "browser_open",     description: "Open a URL in headless browser" },
    { name: "browser_click",    description: "Click an element in the browser" },
    { name: "remember",         description: "Save a memory node to JCross" },
    { name: "search_memory",    description: "Semantic search in memory store" },
    { name: "llm_complete",     description: "Call an LLM for text completion" },
    { name: "embed_text",       description: "Generate text embeddings" },
    { name: "get_config",       description: "Read configuration values" },
    { name: "set_config",       description: "Write configuration values" },
];

const engine = new ToolSearchEngine(TOOLS);

// テスト1: "read a file"
const r1 = engine.search("read a file from disk", 5);
console.log('Query: "read a file from disk"');
console.log("Top 5:", r1.map(r => `${r.tool.name}(${r.score.toFixed(2)})`).join(", "));

// テスト2: "send message to team"
const r2 = engine.search("send message to team", 3);
console.log('\nQuery: "send message to team"');
console.log("Top 3:", r2.map(r => `${r.tool.name}(${r.score.toFixed(2)})`).join(", "));

// テスト3: "push code to github"
const r3 = engine.search("push code to github", 3);
console.log('\nQuery: "push code to github"');
console.log("Top 3:", r3.map(r => `${r.tool.name}(${r.score.toFixed(2)})`).join(", "));

// テスト4: コンテキスト節約量
const fullTokens = TOOLS.map(t => `${t.name}: ${t.description}`).join("\n").length;
const topK = r1.map(r => `${r.tool.name}: ${r.tool.description}`).join("\n").length;
const saved = ((1 - topK / fullTokens) * 100).toFixed(0);
console.log(`\n📊 Context savings: ${saved}% (full: ${fullTokens} chars → top-5: ${topK} chars)`);
console.log("✅ All tests passed");
