// Builds presentation/MatGraphRAG_Review.pptx in the style of the first-review Beamer deck
// (4:3, blue frame titles, three-part footer), with architecture, results and demo slides added.
// Needs pptxgenjs, jszip, react, react-dom, react-icons and sharp on NODE_PATH.
// Usage: NODE_PATH=<node_modules> node presentation/build_review_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { fixParagraphs, applyThemeColours } = require("./pptx_fixups");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const HERE = __dirname;
const ASSET = (f) => path.join(HERE, "assets", f);
const OUT = path.join(HERE, "MatGraphRAG_Review.pptx");

const FOOT_LEFT = "Group 14 (Amrita Vishwa Vidyapeetham)";
const DATE = "October 23, 2026";
const TOTAL = 23;

// Beamer default blues for structure; draw.io-style stroke colours for diagrams.
const THEME = {
  name: "MatGraphRAG Review",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "000000",
    lt1: "FFFFFF",
    dk2: "191959", // footer left, dark structure
    lt2: "EBEBF7", // block body
    accent1: "3333B2", // beamer structure blue
    accent2: "262686", // footer middle
    accent3: "82B366", // diagram green
    accent4: "D79B00", // diagram orange
    accent5: "6C8EBF", // diagram blue
    accent6: "9673A6", // diagram purple
    hlink: "3333B2",
    folHlink: "262686",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_4x3"; // 10 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "MatGraphRAG";
pres.subject = "MatGraphRAG review";
const C = pres.SchemeColor;
// diagram palette: [stroke, fill transparency]
const D = { blue: C.accent5, green: C.accent3, orange: C.accent4, purple: C.accent6, red: C.accent1 };

async function icon(name, color = "#FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { color, size: String(size) }));
  return "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");
}

// ---------- layouts ----------
const footer = [
  { rect: { x: 0, y: 7.18, w: 3.334, h: 0.32, fill: { color: C.text2 }, line: { type: "none" } } },
  { rect: { x: 3.334, y: 7.18, w: 3.333, h: 0.32, fill: { color: C.accent2 }, line: { type: "none" } } },
  { rect: { x: 6.667, y: 7.18, w: 3.333, h: 0.32, fill: { color: C.accent1 }, line: { type: "none" } } },
  { text: { text: FOOT_LEFT, options: { x: 0, y: 7.18, w: 3.334, h: 0.32, fontSize: 10, color: C.background1, align: "center", valign: "middle", margin: 0 } } },
  { text: { text: "MatGraphRAG", options: { x: 3.334, y: 7.18, w: 3.333, h: 0.32, fontSize: 10, color: C.background1, align: "center", valign: "middle", margin: 0 } } },
  { text: { text: DATE, options: { x: 6.75, y: 7.18, w: 2.0, h: 0.32, fontSize: 10, color: C.background1, align: "center", valign: "middle", margin: 0 } } },
  { text: { text: `/ ${TOTAL}`, options: { x: 9.23, y: 7.18, w: 0.6, h: 0.32, fontSize: 10, color: C.background1, valign: "middle", margin: 0 } } },
];
const slideNumber = { x: 8.68, y: 7.215, w: 0.5, h: 0.25, fontSize: 10, color: C.background1, align: "right" };

pres.defineSlideMaster({ title: "TITLE", background: { color: C.background1 }, objects: footer, slideNumber });
pres.defineSlideMaster({
  title: "FRAME",
  background: { color: C.background1 },
  margin: [0.95, 0.4, 0.5, 0.4],
  objects: [
    { rect: { x: 0, y: 0, w: 10, h: 0.78, fill: { color: C.accent1 }, line: { type: "none" } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.22, y: 0, w: 9.6, h: 0.78, fontSize: 26, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    ...footer,
  ],
  slideNumber,
});

// ---------- text helpers ----------
let objN = 0;
const nm = (s) => `${s}-${++objN}`;

// Inline markup: **bold**, `code`, {{blue bold}}.
function runs(str, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\{\{[^}]+\}\})/g;
  let last = 0, m;
  while ((m = re.exec(str))) {
    if (m.index > last) out.push({ text: str.slice(last, m.index), options: { ...base } });
    const t = m[0];
    if (t.startsWith("**")) out.push({ text: t.slice(2, -2), options: { ...base, bold: true } });
    else if (t.startsWith("`")) out.push({ text: t.slice(1, -1), options: { ...base, fontFace: "Courier New" } });
    else out.push({ text: t.slice(2, -2), options: { ...base, bold: true, color: C.accent1 } });
    last = m.index + t.length;
  }
  if (last < str.length) out.push({ text: str.slice(last), options: { ...base } });
  return out;
}

// items: "text" (level 0), [1, "text"] (level 1), {h: "Heading"} (bold, no bullet), {p: "text"} (no bullet)
function bullets(s, items, box) {
  const size = box.fontSize || 18;
  const paras = [];
  // pptxgenjs starts a new paragraph at every run carrying `bullet`, so paragraph options go on the first run only.
  items.forEach((it) => {
    let p;
    if (typeof it === "string") {
      p = runs(it, { fontSize: size });
      Object.assign(p[0].options, { bullet: { code: "25CF", indent: size * 1.1 } });
    } else if (Array.isArray(it)) {
      p = runs(it[1], { fontSize: size - 2 });
      Object.assign(p[0].options, { bullet: { code: "25CF", indent: size * 0.95 }, indentLevel: 1 });
    } else if (it.p) {
      p = runs(it.p, { fontSize: size });
      p[0].options.paraSpaceBefore = it.space ?? 6;
    } else {
      p = runs(it.h, { bold: true, fontSize: size });
      p[0].options.paraSpaceBefore = it.space ?? 6;
    }
    paras.push(p);
  });
  const all = [];
  paras.forEach((p, i) => p.forEach((r, j) => {
    if (i < paras.length - 1 && j === p.length - 1) r.options.breakLine = true;
    all.push(r);
  }));
  s.addText(all, Object.assign({ isTextBox: true, margin: 0, color: C.text1, valign: "top", paraSpaceAfter: box.gap ?? 7, objectName: nm("bullets") }, box, { fontSize: size }));
}

function text(s, t, o) {
  s.addText(typeof t === "string" ? runs(t) : t, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", objectName: nm("text") }, o));
}

function frame(title) {
  const s = pres.addSlide({ masterName: "FRAME" });
  s.addText(title, { placeholder: "title" });
  return s;
}

// ---------- shape helpers ----------
function box(s, x, y, w, h, stroke, o = {}) {
  s.addShape(o.shape || pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: o.radius ?? 0.08, objectName: nm(o.name || "box"),
    fill: { color: o.fill || stroke, transparency: o.fill ? 0 : (o.tint ?? 82) },
    line: { color: stroke, width: o.lw || 1.25, dashType: o.dash || "solid" },
  });
}

function boxText(s, x, y, w, h, stroke, head, sub, o = {}) {
  box(s, x, y, w, h, stroke, o);
  const parts = [{ text: head, options: { bold: true, fontSize: o.headSize || 13, breakLine: !!sub } }];
  if (sub) parts.push({ text: sub, options: { fontSize: o.subSize || 11, color: C.text1 } });
  text(s, parts, { x: x + 0.08, y: y + 0.04, w: w - 0.16, h: h - 0.08, align: o.align || "center", valign: "middle" });
}

function line(s, x1, y1, x2, y2, o = {}) {
  s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, objectName: nm("edge"),
    line: { color: o.color || "555555", width: o.width || 1.5, dashType: o.dash || "solid", endArrowType: o.arrow === false ? undefined : "triangle" },
  });
}

function pill(s, label, x, y, w, h, color, o = {}) {
  s.addText(label, {
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: h / 2, x, y, w, h,
    fill: { color, transparency: o.tint || 0 }, line: o.line ? { color: o.line, width: 1.25, dashType: o.dash || "solid" } : { type: "none" },
    fontSize: o.fontSize || 12, bold: o.bold ?? true, color: o.color || C.background1, align: "center", valign: "middle", margin: 0, objectName: nm("pill"),
  });
}

function numDot(s, n, x, y, d, color, size = 13) {
  s.addText(String(n), { shape: pres.shapes.OVAL, x, y, w: d, h: d, fill: { color }, line: { type: "none" }, fontSize: size, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, objectName: nm("num") });
}

function block(s, title, body, x, y, w, h) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 0.38, fill: { color: C.accent1 }, line: { type: "none" }, objectName: nm("block-head") });
  s.addShape(pres.shapes.RECTANGLE, { x, y: y + 0.38, w, h: h - 0.38, fill: { color: C.background2 }, line: { type: "none" }, objectName: nm("block-body") });
  text(s, title, { x: x + 0.12, y, w: w - 0.24, h: 0.38, fontSize: 16, color: C.background1, valign: "middle" });
  text(s, body, { x: x + 0.12, y: y + 0.45, w: w - 0.24, h: h - 0.5, fontSize: 14 });
}

const TB = { type: "solid", pt: 0.75, color: "9A9AC8" };
function table(s, rows, o) {
  s.addTable(rows, Object.assign({ fontFace: "Arial", fontSize: 12, color: C.text1, border: TB, valign: "middle", margin: [0.03, 0.06, 0.03, 0.06], objectName: nm("table") }, o));
}
const th = (t, o = {}) => ({ text: t, options: { bold: true, fill: { color: C.background2 }, ...o } });
const mono = (t, o = {}) => ({ text: t, options: { fontFace: "Courier New", ...o } });

// ======================================================================
async function build() {
  // ---------- 1. Title ----------
  {
    const s = pres.addSlide({ masterName: "TITLE" });
    s.addText([
      { text: "MatGraphRAG: An Explainable Graph Based Retrieval Augmented Reasoning System for Intelligent Materials Discovery", options: { fontSize: 24, breakLine: true } },
      { text: "Second Review — Phases 1–3: Knowledge Graph Construction, GraphRAG Retrieval & Crystal Structure Encoding", options: { fontSize: 16 } },
    ], {
      shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.12, x: 0.45, y: 0.35, w: 9.1, h: 2.2, fill: { color: C.accent1 }, line: { type: "none" },
      shadow: { type: "outer", color: "9A9A9A", blur: 4, offset: 3, angle: 90, opacity: 0.5 }, color: C.background1, align: "center", valign: "middle",
      margin: 0.15, paraSpaceAfter: 6, objectName: "title-box",
    });
    text(s, [
      { text: "Group – 7", options: { breakLine: true } },
      { text: "Dhakshin – CB.SC.U4AIE24313", options: { breakLine: true } },
      { text: "Naga Shiva – CB.SC.U4AIE24315", options: { breakLine: true } },
      { text: "Rajashree T – CB.SC.U4AIE24346", options: { breakLine: true } },
      { text: "Sharvesh – CB.SC.U4AIE24355" },
    ], { x: 1, y: 2.95, w: 8, h: 1.95, fontSize: 18, align: "center", valign: "middle", paraSpaceAfter: 2 });
    text(s, "Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore", { x: 0.5, y: 5.15, w: 9, h: 0.4, fontSize: 13, align: "center", valign: "middle" });
    text(s, DATE, { x: 1, y: 5.95, w: 8, h: 0.45, fontSize: 18, align: "center", valign: "middle" });
  }

  // ---------- 2. Introduction ----------
  {
    const s = frame("Introduction");
    bullets(s, [
      "**Materials** form the foundation of modern technologies — batteries, semiconductors, solar cells and electronic devices.",
      "Their **composition and crystal structure** determine the properties that govern performance.",
      "Finding materials with suitable properties is therefore crucial for designing real-world technologies.",
      "The search space is enormous — the Materials Project indexes **154,377** inorganic materials.",
      "Materials, properties, structures and applications are deeply **interrelated**, yet current tools store them as isolated rows in a table.",
    ], { x: 0.55, y: 1.1, w: 8.9, h: 5.8, fontSize: 20, valign: "middle", gap: 10 });
  }

  // ---------- 3. Problem statement ----------
  {
    const s = frame("Problem Statement");
    bullets(s, [
      "Existing material search tools return **raw data** or **text-similarity matches**.",
      "They do not reason about the real relationships between **materials**, **properties**, and **applications**.",
      "Recommendations, when produced, are **not traceable** to any underlying evidence.",
      "**Consequences:**",
      [1, "Researchers must manually cross-reference multiple databases."],
      [1, "LLM-based answers risk hallucinating unsupported material claims."],
      [1, "No audit trail from a recommendation back to the source record."],
      "**This project** builds a materials knowledge graph and a GraphRAG-based retrieval system that answers natural-language queries with **explainable, evidence-traceable** recommendations.",
    ], { x: 0.55, y: 1.0, w: 8.9, h: 6.0, fontSize: 19, valign: "middle", gap: 7 });
  }

  // ---------- 4. Objectives ----------
  {
    const s = frame("Objectives");
    bullets(s, [
      "**Build (Phase 1):** construct a Materials Knowledge Graph from real Materials Project data using deterministic, rule-based classification — fully auditable, zero LLM calls.",
      "**Retrieve (Phase 2):** implement GraphRAG relationship-aware retrieval — parse a natural-language question into structured constraints and traverse the graph.",
      "**Explain (Phase 2):** generate recommendations grounded only in retrieved graph evidence — every claim traceable to a specific edge or record.",
      "**Encode (Phase 3):** integrate CGCNN crystal-structure embeddings for structure-aware retrieval and elemental substitutability.",
      "**Scale:** migrate to Neo4j and expose the system through a backend API and visualization dashboard.",
    ], { x: 0.55, y: 1.0, w: 8.9, h: 6.0, fontSize: 18, valign: "middle", gap: 9 });
  }

  // ---------- 5-6. Literature review ----------
  const litHead = [th("#"), th("Year"), th("Title"), th("Technique"), th("Inference"), th("Limitations")];
  const lit = [
    ["1", "2024", "LLaMP: Large Language Model Made Powerful for High Fidelity Materials Knowledge Retrieval", "Multi agent LLM system querying trusted scientific databases", "Retrieves materials knowledge and generates grounded responses", "Document retrieval only; no graph based reasoning"],
    ["2", "2024", "HoneyComb: A Flexible LLM Based Agent System for Materials Science", "Extensible LLM agent framework integrating scientific tools", "Flexible tool-augmented materials research assistant", "No explicit knowledge graph; no relationship aware retrieval"],
    ["3", "2024", "LLaMat: Foundational Large Language Models for Materials Research", "Materials specific foundation model trained on literature", "Strong domain specific language understanding", "Knowledge is static; no structured graph reasoning"],
    ["4", "2024", "From Local to Global: A Graph RAG Approach to Query Focused Summarization", "Entity extraction, graph construction, multi hop retrieval", "Graph structure improves multi hop reasoning over text", "Generic framework; not designed for materials science"],
    ["5", "2018", "Crystal Graph Convolutional Neural Networks (CGCNN)", "Atoms as nodes, bonds as edges; graph convolution + pooling", "Accurate property prediction from crystal structure", "Prediction only; no retrieval or reasoning"],
    ["6", "2021", "ALIGNN: Atomistic Line Graph Neural Network", "Line graph encoding of bond angles", "Improved property prediction over CGCNN", "No integration with knowledge graphs or RAG"],
    ["7", "2013", "The Materials Project: A materials genome approach", "Open DFT computed database, 154,377 materials", "Large-scale structured property data freely accessible", "Raw database; no reasoning or recommendation layer"],
  ];
  const litW = [0.32, 0.55, 2.35, 1.95, 1.95, 2.0];
  {
    const s = frame("Literature Review");
    table(s, [litHead, ...lit.slice(0, 4)], { x: 0.44, y: 1.2, w: 9.12, colW: litW, fontSize: 13.5 });
  }
  {
    const s = frame("Literature Review (Cont.)");
    const ours = ["Ours", "MatGraphRAG", "Materials KG + GraphRAG retrieval + CGCNN embeddings", "Explainable, evidence-traceable recommendation", "—"]
      .map((t, i) => ({ text: t, options: { bold: true, fill: { color: C.accent1, transparency: 80 }, colspan: i === 0 ? 2 : undefined } }));
    table(s, [litHead, ...lit.slice(4), ours], { x: 0.44, y: 1.05, w: 9.12, colW: litW, fontSize: 13 });
    block(s, "Research Gap", [
      { text: "No prior system combines a " }, { text: "materials knowledge graph", options: { bold: true } }, { text: ", " },
      { text: "relationship-aware GraphRAG retrieval", options: { bold: true } }, { text: ", and " },
      { text: "structure-aware CGCNN encoding", options: { bold: true } }, { text: " into a single explainable discovery pipeline." },
    ], 0.44, 5.85, 9.12, 1.15);
  }

  // ---------- 7. Research gap and novelty ----------
  {
    const s = frame("Research Gap and Novelty");
    bullets(s, [
      { h: "Research Gap:", space: 0 },
      "**Material databases** make data available, but only as isolated records.",
      "**Traditional RAG** retrieves by text similarity, not by real relationships.",
      "**Knowledge graphs** encode relationships but are not connected to LLM reasoning.",
      "**CGCNN / ALIGNN** use crystal structure, but only for property prediction.",
      { h: "Novelty of This Work:", space: 10 },
      "First system integrating a materials knowledge graph + GraphRAG retrieval + CGCNN structural encoding.",
      "**Deterministic construction** — rule-based, zero LLM calls, so every node and edge is auditable and reproducible.",
      "**Explainability built into the edge itself** — each `suitable_for` edge stores the exact rule text that fired.",
      "**Six grouping axes** (element, chemical system, symmetry, property bucket, oxidation state, formula pattern) so materials relate even with zero shared elements.",
    ], { x: 0.45, y: 0.95, w: 9.1, h: 6.1, fontSize: 16, valign: "middle", gap: 4 });
  }

  // ---------- 8. Dataset ----------
  {
    const s = frame("Dataset — Materials Project");
    bullets(s, [
      { p: "**Source:** Materials Project REST API — computed properties plus human-readable structure descriptions. MP indexes **154,377** materials in total.", space: 0 },
      { h: "Why not the full catalog?", space: 10 },
      "Material–material similarity is **quadratic** — 154,377 materials means ∼11.9 billion pairs.",
      "An exhaustive pull produces a **sparse, disconnected** graph; Phase 1’s goal is a densely interconnected one.",
      { p: "**Sampling — 8 chemistry-driven clusters:** battery cathodes (oxide, phosphate), semiconductors (III–V, II–VI, group IV), binary oxides, perovskite oxides, magnetic spinel oxides. Stability cutoff E_hull ≤ 0.08 eV/atom; radioactive and synthetic elements excluded.", space: 10 },
      { p: "**Result:** 8 clusters → ∼2,000 raw candidates → **794 unique materials** after deduplication by formula.", space: 10 },
      { p: "**Properties:** band gap, formation energy, energy above hull, density, magnetization and symmetry at 100% coverage; oxidation states 93%; elastic 18%; dielectric 20%.", space: 10 },
    ], { x: 0.45, y: 0.95, w: 9.1, h: 6.1, fontSize: 16, valign: "middle", gap: 4 });
  }

  // ---------- 9. Methodology overview ----------
  {
    const s = frame("Methodology — Overview");
    bullets(s, [
      { h: "Proposed System Architecture:", space: 0 },
      "**Materials Project API** → material properties.",
      "**Knowledge graph construction** — entities and relationships.",
      "**GraphRAG retrieval** — relationship-aware multi-hop traversal.",
      "**LLM reasoning** — grounded generation over retrieved evidence.",
      "**Explainable material recommendation** — final output.",
    ], { x: 0.45, y: 1.2, w: 9.1, h: 3.1, fontSize: 18, gap: 5 });
    text(s, "Three implementation phases:", { x: 0.45, y: 4.55, w: 9.1, h: 0.4, fontSize: 18, bold: true });
    const phases = [
      ["Phase 1", "Knowledge Graph Construction", "[COMPLETE]"],
      ["Phase 2", "GraphRAG Retrieval and Recommendation", "[COMPLETE]"],
      ["Phase 3", "Crystal Structure Encoding", "[COMPLETE]"],
    ];
    phases.forEach(([p, d, st], i) => {
      const y = 5.05 + i * 0.5;
      text(s, [{ text: "●  ", options: { color: C.accent1 } }, { text: p, options: { bold: true } }, { text: " — " + d }], { x: 0.75, y, w: 6.6, h: 0.42, fontSize: 17, valign: "middle" });
      text(s, st, { x: 7.3, y, w: 2.25, h: 0.42, fontSize: 17, bold: true, color: C.accent1, align: "right", valign: "middle" });
    });
    s.addNotes("All three phases are complete: the knowledge graph, GraphRAG retrieval with grounded explanations, and crystal structure encoding with the pre-trained CGCNN models wired into material ingestion. The following slides add the architecture diagrams.");
  }

  // ---------- 10. Overall system architecture (new) ----------
  {
    const s = frame("Methodology — Overall System Architecture");
    const bands = [
      [D.green, "Data sources", [["Materials Project API", "summary + robocrys"], ["Cached raw JSON", "794 records"], ["User input", "MP id · formula · CIF"], ["materials.yaml", "clusters · buckets · rules"]]],
      [D.blue, "Knowledge graph", [["Build pipeline", "fetch → enrich → write"], ["Cognee DataPoints", "identity-keyed nodes"], ["Kuzu + LanceDB", "graph + vector stores"], ["BGE-M3 embeddings", "local Ollama, 1024-d"]]],
      [D.purple, "Reasoning services", [["Query parser", "LLM + rule fallback"], ["Graph retriever", "traverse · filter · rank"], ["Evidence → grounding", "cited, checked claims"], ["Material ingestion", "MP · formula · CGCNN"]]],
      [D.orange, "Interfaces", [["FastAPI service", "/query · /ingest · /graph"], ["Dashboard", "ask · add · structure"], ["Graph explorer", "import & match"], ["CLI scripts", "build · ask · evaluate"]]],
    ];
    const bh = 1.18, gap = 0.3, y0 = 1.0, lw = 1.55, cx0 = 0.4 + lw + 0.15, cw = (9.6 - cx0 - 3 * 0.12) / 4;
    bands.forEach(([col, name, comps], i) => {
      const y = y0 + i * (bh + gap);
      box(s, 0.4, y, 9.2, bh, col, { tint: 90, name: "layer" });
      text(s, name, { x: 0.5, y, w: lw - 0.1, h: bh, fontSize: 14, bold: true, valign: "middle" });
      comps.forEach(([h, sub], j) => boxText(s, cx0 + j * (cw + 0.12), y + 0.14, cw, bh - 0.28, col, h, sub, { fill: C.background1, headSize: 12, subSize: 10.5 }));
      if (i < bands.length - 1) line(s, 5.0, y + bh, 5.0, y + bh + gap, { color: "555555", width: 2 });
    });
    s.addNotes("The system in four layers, top to bottom: data sources, the knowledge graph, reasoning services, and the interfaces that expose them. Every service reads the same Cognee-managed graph and vector stores.");
  }

  // ---------- 11. Phase 1 construction pipeline ----------
  {
    const s = frame("Methodology — Phase 1 Construction Pipeline");
    bullets(s, [
      "**1. FETCH** (`mp_client.py`) — one query per cluster pattern, sorted by stability; deduplicate by formula; cache to `data/raw/` for reproducibility.",
      "**2. ENRICH** (`enrich.py`) — **no LLM**: classify band gap / stability / magnetic ordering into buckets, apply application-domain rules, compute `similar_to` neighbours.",
      "**3. WRITE** (`pipeline.py`) — Cognee pipeline writes the graph store and vector store together, deduplicating on each node’s identity key.",
      "**4. VERIFY** (`verify.py`) — node/edge counts and structural queries read directly from the graph store; renders an interactive visualization.",
      { h: "Grouping mechanisms (all deterministic):", space: 8 },
      "**Buckets:** band gap (metal / narrow / semiconductor / wide), stability (stable / metastable / unstable), magnetic ordering.",
      "**Domain rules:** battery cathode, semiconductor device, photocatalyst, dielectric, magnetic material — each rule’s text stored on the edge.",
      "**Categorical fields:** oxidation states and formula patterns group directly, no thresholds needed.",
    ], { x: 0.45, y: 0.95, w: 9.1, h: 6.1, fontSize: 15, valign: "middle", gap: 4 });
  }

  // ---------- 12. Phase 1 architecture (redrawn) ----------
  {
    const s = frame("Methodology — Phase 1 Architecture");
    const X = 0.45, W = 2.3, H = 0.72, G = 0.3;
    const spine = [
      [D.blue, "Materials Project API", "154,377 inorganic materials"],
      [D.green, "1 · FETCH", "cluster queries · dedupe by formula"],
      [D.orange, "794 cached records", "properties + crystal structures"],
      [D.green, "2 · ENRICH  (no LLM)", "buckets · domain rules · similar_to"],
      [D.green, "3 · BUILD GRAPH", "Cognee pipeline → Kuzu + LanceDB"],
    ];
    spine.forEach(([col, h, sub], i) => {
      const y = 1.05 + i * (H + G);
      boxText(s, X, y, W, H, col, h, sub, { shape: i === 2 ? pres.shapes.CAN : undefined, headSize: 12, subSize: 10 });
      if (i < spine.length - 1) line(s, X + W / 2, y + H, X + W / 2, y + H + G, { width: 1.75 });
    });
    const yBuild = 1.05 + 4 * (H + G);
    // knowledge graph
    const kx = 3.25, ky = 2.65, kw = 4.0, kh = 3.72;
    box(s, kx, ky, kw, kh, D.green, { lw: 2.25, tint: 88, name: "knowledge-graph" });
    text(s, [
      { text: "MATERIALS KNOWLEDGE GRAPH", options: { bold: true, fontSize: 14, breakLine: true } },
      { text: "1,504 nodes · 14,686 edges", options: { fontSize: 13, color: C.accent1, bold: true } },
    ], { x: kx + 0.1, y: ky + 0.12, w: kw - 0.2, h: 0.65, align: "center" });
    const types = ["Material 794", "FormulaPattern 261", "ChemicalSystem 143", "OxidationState 131", "SpaceGroup 106", "Element 46", "PropertyClass 11", "CrystalSystem 7", "ApplicationDomain 5"];
    types.forEach((t, i) => {
      const c = i % 2, r = Math.floor(i / 2);
      pill(s, t, kx + 0.2 + c * 1.85, ky + 0.95 + r * 0.52, 1.75, 0.38, i === 0 ? C.accent1 : D.green, { fontSize: 10.5, tint: i === 0 ? 0 : 0 });
    });
    line(s, X + W, yBuild + H / 2, kx, yBuild + H / 2, { width: 1.75 });
    // config note
    boxText(s, 3.25, 1.05, 2.0, 0.85, C.accent4, "materials.yaml", "8 clusters · buckets · rules", { headSize: 12, subSize: 10, tint: 80, dash: "dash" });
    line(s, 3.25, 1.47, X + W, 1.05 + (H + G) + H / 2, { dash: "dash", width: 1.25 });
    text(s, "configures", { x: 3.0, y: 2.0, w: 1.0, h: 0.24, fontSize: 10, italic: true, color: C.text2 });
    // verify + explainability
    boxText(s, 7.6, 2.65, 1.95, 0.95, D.green, "4 · VERIFY", "counts · structural checks", { headSize: 12, subSize: 10 });
    line(s, kx + kw, 3.12, 7.6, 3.12, { width: 1.5 });
    boxText(s, 7.6, 3.95, 1.95, 0.95, D.blue, "graph.html", "interactive visualization", { headSize: 12, subSize: 10 });
    line(s, 8.575, 3.6, 8.575, 3.95, { width: 1.5 });
    boxText(s, 7.6, 5.3, 1.95, 1.07, C.accent4, "Explainability", "each suitable_for edge stores the rule text that fired", { headSize: 12, subSize: 10, tint: 80, dash: "dash" });
    line(s, 7.6, 5.84, kx + kw, 5.84, { dash: "dash", width: 1.25 });
    text(s, "Zero LLM calls · every node and edge reproducible from the local cache", { x: 0.45, y: 6.55, w: 9.1, h: 0.35, fontSize: 13, italic: true, color: C.text2, align: "center", valign: "middle" });
    s.addNotes("The Phase 1 build as a diagram. Materials Project data is fetched per cluster, cached, enriched by deterministic rules and written through Cognee into the knowledge graph, which is then verified and visualised.");
  }

  // ---------- 13. Graph schema tables ----------
  {
    const s = frame("Methodology — Graph Schema");
    text(s, "Node types — 1,504", { x: 0.5, y: 1.05, w: 4.1, h: 0.4, fontSize: 17, bold: true, align: "center" });
    text(s, "Relationship types", { x: 5.0, y: 1.05, w: 4.5, h: 0.4, fontSize: 17, bold: true, align: "center" });
    const nodes = [["Material", "794"], ["FormulaPattern", "261"], ["ChemicalSystem", "143"], ["OxidationState", "131"], ["SpaceGroup", "106"], ["Element", "46"], ["PropertyClass", "11"], ["CrystalSystem", "7"], ["ApplicationDomain", "5"]];
    const edges = [["contains", "—"], ["has_space_group", "—"], ["member_of", "—"], ["classified_as", "—"], ["suitable_for", "rule: fired text"], ["similar_to", "weight: cosine"], ["crystal_system", "—"], ["has_oxidation_state", "—"], ["has_formula_pattern", "—"]];
    table(s, [[th("Type"), th("Count", { align: "right" })], ...nodes.map(([a, b]) => [mono(a), { text: b, options: { align: "right" } }])],
      { x: 0.5, y: 1.5, w: 4.1, colW: [2.9, 1.2], rowH: 0.32, fontSize: 12.5 });
    table(s, [[th("Edge"), th("Properties")], ...edges.map(([a, b]) => [mono(a), { text: b }])],
      { x: 5.0, y: 1.5, w: 4.5, colW: [2.5, 2.0], rowH: 0.32, fontSize: 12.5 });
    text(s, "**Design principle:** continuous values stay as **attributes**; anything used for **grouping** becomes a **node**; derived material–material relations become **weighted edges**. `similar_to` uses top-5 nearest neighbours by cosine similarity over a 10-dimensional z-scored property vector.",
      { x: 0.5, y: 5.0, w: 9.0, h: 1.4, fontSize: 15 });
  }

  // ---------- 14. Graph schema diagram (new) ----------
  {
    const s = frame("Methodology — Graph Schema Diagram");
    const cx = 5.0, cy = 3.95;
    const nodes = [
      ["Oxidation state · 131", 1.75, 1.6, 2.3, D.blue, "has_oxidation_state", [1.3, 2.72, 2.1, "right"]],
      ["Chemical system · 143", 5.0, 1.35, 2.3, D.green, "member_of", [5.12, 2.4, 1.3, "left"]],
      ["Formula pattern · 261", 8.25, 1.6, 2.3, D.purple, "has_formula_pattern", [6.6, 2.72, 2.1, "left"]],
      ["Element · 46", 1.3, 3.95, 1.7, D.green, "contains", [2.15, 3.6, 1.75, "center"]],
      ["Space group · 106", 8.45, 3.95, 2.1, D.purple, "has_space_group", [6.1, 3.6, 1.3, "center"]],
      ["Property class · 11", 1.75, 6.35, 2.3, D.orange, "classified_as", [1.3, 5.05, 1.85, "right"]],
      ["Application domain · 5", 5.0, 6.5, 2.6, C.accent1, "suitable_for + rule", [5.12, 5.05, 1.7, "left"]],
      ["Material", 8.25, 6.35, 2.0, C.accent2, "similar_to (weight)", [5.3, 5.62, 1.75, "right"]],
    ];
    for (const [, x, y] of nodes) line(s, cx, cy, x, y, { arrow: false, color: "666666", width: 1.25 });
    line(s, 8.45, 3.95, 8.45, 5.15, { arrow: false, color: "666666", width: 1.25 });
    for (const [lab, x, y, w, col, rel, [lx, ly, lw, al]] of nodes) {
      pill(s, lab, x - w / 2, y - 0.25, w, 0.5, col, { fontSize: 12 });
      text(s, rel, { x: lx, y: ly, w: lw, h: 0.26, fontSize: 11, italic: true, color: C.text2, align: al, valign: "middle" });
    }
    pill(s, "Crystal system · 7", 8.45 - 1.05, 5.15 - 0.25, 2.1, 0.5, D.purple, { fontSize: 12 });
    text(s, "crystal_system", { x: 7.05, y: 4.42, w: 1.3, h: 0.26, fontSize: 11, italic: true, color: C.text2, align: "right", valign: "middle" });
    pill(s, "Material · 794", cx - 1.1, cy - 0.38, 2.2, 0.76, C.accent1, { fontSize: 17 });
    s.addNotes("The schema as a graph. Material is the centre; every grouping dimension is a shared hub node, space groups roll up into crystal systems, and similar_to links materials directly with a cosine weight.");
  }

  // ---------- 15. Phase 2 GraphRAG pipeline (new) ----------
  {
    const s = frame("Methodology — Phase 2 GraphRAG Pipeline");
    s.addText([{ text: "Q  ", options: { bold: true, color: C.accent4 } }, { text: "stable, cobalt-free Li cathode materials with a band gap above 2 eV", options: { color: C.background1 } }], {
      shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x: 0.4, y: 0.98, w: 9.2, h: 0.48, fill: { color: C.text2 }, line: { type: "none" }, fontSize: 14, valign: "middle", margin: [0, 0.15, 0, 0.15], objectName: "question",
    });
    const steps = [
      [C.accent1, "Parse", "LLM (JSON schema) → typed constraints; rule-based fallback; values checked against graph vocabulary", "battery_cathode · band_gap ≥ 2 · stable · contains Li · no Co"],
      [D.green, "Retrieve", "Seed constraint nodes, traverse back to materials, intersect, enforce numeric ranges, rank", "39 of 77 reached materials pass every constraint; top: LiScO2"],
      [D.blue, "Evidence", "Flatten into atomic facts [E1] [E2] …: an MP field or a graph edge with its rule text", "[E4] LiScO2 band_gap = 3.904 eV"],
      [D.purple, "Explain", "LLM writes claims citing fact ids, or a templated explanation without an LLM", "“LiScO2 is a stable cathode candidate … [E4]”"],
      [D.orange, "Ground", "Checker: ids exist, every number matches a cited fact, no unretrieved materials", "69 / 69 claims grounded (100%)"],
    ];
    const rh = 0.86, rg = 0.13, y0 = 1.62;
    steps.forEach(([col, h, d, ex], i) => {
      const y = y0 + i * (rh + rg);
      pill(s, `${i + 1}  ${h}`, 0.4, y + 0.18, 1.55, 0.5, col, { fontSize: 14 });
      text(s, d, { x: 2.1, y, w: 4.3, h: rh, fontSize: 12.5, valign: "middle" });
      box(s, 6.55, y + 0.04, 3.05, rh - 0.08, col, { tint: 88, name: "example" });
      text(s, ex, { x: 6.65, y: y + 0.04, w: 2.85, h: rh - 0.08, fontSize: 11.5, bold: true, valign: "middle" });
      if (i < steps.length - 1) line(s, 1.175, y + 0.68, 1.175, y + rh + rg + 0.18, { width: 1.5 });
    });
    text(s, [{ text: "The LLM never selects or filters materials. ", options: { bold: true, color: C.accent1 } }, { text: "Numeric limits, exclusions and rules are enforced exactly on graph attributes and edges." }],
      { x: 0.4, y: 6.6, w: 9.2, h: 0.45, fontSize: 13, valign: "middle" });
    s.addNotes("Phase 2: the LLM only parses the question and phrases the explanation. Retrieval enforces every constraint exactly; evidence becomes numbered facts; a deterministic checker verifies every citation and number.");
  }

  // ---------- 16. Ingestion architecture (new) ----------
  {
    const s = frame("Methodology — New Material Ingestion");
    const steps = [
      [D.green, "Input", ["MP id, formula, CIF or POSCAR", "dashboard, API or CLI"]],
      [D.orange, "Property source", ["MP record → mp", "formula → derived", "CGCNN → predicted ± error"]],
      [D.purple, "Deduplicate", ["formula → MP node", "StructureMatcher", "else new struct-… id"]],
      [C.accent1, "Build nodes", ["same NodeFactory", "identity keys reuse hubs", "only missing hubs added"]],
      [D.blue, "Relate", ["hubs by selectivity", "similar_to (frozen scale)", "compositional neighbours"]],
      [D.purple, "Relation report", ["properties + provenance", "rules fired / not fired / cannot evaluate"]],
      [C.accent1, "Explain & ground", ["LLM or template", "claims checked against facts"]],
      [D.green, "Preview / persist", ["in-memory overlay", "or saved, replayed on rebuild"]],
    ];
    const cw = 2.12, gap = 0.24, h = 2.15, rows = [1.0, 3.65];
    steps.forEach(([col, head, lines], i) => {
      const r = Math.floor(i / 4), c = i % 4, x = 0.4 + c * (cw + gap), y = rows[r];
      box(s, x, y, cw, h, col, { fill: C.background1, lw: 1.5, name: "ingest-step" });
      numDot(s, i + 1, x + 0.1, y + 0.13, 0.38, col, 12);
      text(s, head, { x: x + 0.55, y: y + 0.1, w: cw - 0.6, h: 0.44, fontSize: 12.5, bold: true, valign: "middle" });
      text(s, lines.map((l, k) => ({ text: l, options: { bullet: { code: "25CF", indent: 9 }, breakLine: k < lines.length - 1 } })),
        { x: x + 0.1, y: y + 0.65, w: cw - 0.18, h: h - 0.75, fontSize: 11.5, paraSpaceAfter: 3 });
      if (c < 3) line(s, x + cw, y + h / 2, x + cw + gap, y + h / 2, { width: 1.5 });
    });
    const x4 = 0.4 + 3 * (cw + gap) + cw / 2, x5 = 0.4 + cw / 2, ym = rows[0] + h + 0.25;
    line(s, x4, rows[0] + h, x4, ym, { arrow: false, width: 1.5 });
    line(s, x4, ym, x5, ym, { arrow: false, width: 1.5 });
    line(s, x5, ym, x5, rows[1], { width: 1.5 });
    text(s, "Provenance:", { x: 0.4, y: 6.05, w: 1.2, h: 0.36, fontSize: 12, bold: true, valign: "middle" });
    [["mp", D.green, "solid"], ["derived", D.blue, "solid"], ["guessed", D.purple, "solid"], ["predicted", D.orange, "dash"]].forEach(([l, col, dash], i) =>
      pill(s, l, 1.65 + i * 1.3, 6.06, 1.15, 0.34, col, { fontSize: 11, tint: 85, color: C.text1, line: col, dash }));
    text(s, "Predicted facts are hedged in explanations and drawn dashed.", { x: 6.9, y: 5.98, w: 2.7, h: 0.5, fontSize: 11, color: C.text2, valign: "middle" });
    text(s, "CGCNN (pre-trained, txie-93) supplies properties only when the input is a bare crystal structure.", { x: 0.4, y: 6.55, w: 9.2, h: 0.4, fontSize: 12, italic: true, color: C.text2, valign: "middle" });
    s.addNotes("How a new material is added. The property source depends on the input and every value keeps its provenance. Deduplication resolves to existing nodes where possible. The same NodeFactory as the full build attaches the material to existing hubs, then relating, reporting, explanation and grounding follow; the result is previewed or persisted.");
  }

  // ---------- 17. How a new material connects (new) ----------
  {
    const s = frame("Methodology — How a New Material Connects");
    const cx = 3.95, cy = 3.95;
    const groups = [
      [0.4, 1.0, 2.3, 1.78, D.green, "contains → Element", ["Ta · shared with 51", "Li · shared with 245", "O · shared with 758"], [2.7, 2.2]],
      [0.4, 3.06, 2.3, 1.78, D.blue, "has_oxidation_state", ["Ta⁵⁺ · shared with 38", "Li⁺ · shared with 227", "O²⁻ · shared with 697"], [2.7, 3.95]],
      [0.4, 5.12, 2.3, 1.85, D.orange, "classified_as / suitable_for", ["wide_gap · 145", "stable · 366", "dielectric · 220", "(rule fired)"], [2.7, 5.7]],
      [2.95, 1.0, 2.0, 1.55, D.purple, "has_space_group", ["R3c (#161) · 2", "→ Trigonal"], [cx, 2.55]],
      [2.95, 5.35, 2.0, 1.62, D.purple, "pattern / chem. system", ["ABC3 · shared with 45", "Li-O-Ta · new node"], [cx, 5.35]],
    ];
    for (const [, , , , col, , , [ax, ay]] of groups) line(s, cx, cy, ax, ay, { color: col, width: 1.75 });
    for (const [x, y, w, h, col, head, lines] of groups) {
      box(s, x, y, w, h, col, { tint: 88, name: "hub-group" });
      text(s, head, { x: x + 0.1, y: y + 0.08, w: w - 0.2, h: 0.48, fontSize: 12, bold: true });
      text(s, lines.map((l, k) => ({ text: l, options: { breakLine: k < lines.length - 1 } })), { x: x + 0.1, y: y + 0.6, w: w - 0.2, h: h - 0.65, fontSize: 12, paraSpaceAfter: 2 });
    }
    line(s, cx, cy, 5.2, cy, { color: C.accent1, width: 1.75 });
    box(s, 5.2, 1.0, 2.2, 5.97, C.accent1, { tint: 90, name: "related-materials" });
    text(s, "Related materials", { x: 5.3, y: 1.08, w: 2.0, h: 0.32, fontSize: 13, bold: true, color: C.accent1, valign: "middle" });
    text(s, "similar_to · property cosine", { x: 5.3, y: 1.42, w: 2.0, h: 0.28, fontSize: 10.5, color: C.text2 });
    const rowsT = (arr) => arr.map(([f, w]) => [{ text: f, options: { bold: true } }, { text: w, options: { align: "right", color: C.text2 } }]);
    table(s, rowsT([["Ba3Hf2O7", "0.976"], ["SrTa4O11", "0.967"], ["Na2Ta4O11", "0.956"], ["K2Ta4O11", "0.954"], ["Ba4Hf3O10", "0.953"]]),
      { x: 5.3, y: 1.75, w: 2.0, colW: [1.35, 0.65], rowH: 0.34, fontSize: 11.5, border: { type: "none" }, margin: [0, 0.03, 0, 0.03] });
    text(s, "compositional neighbours", { x: 5.3, y: 3.65, w: 2.0, h: 0.28, fontSize: 10.5, color: C.text2 });
    table(s, rowsT([["Ta3O8", "0.953"], ["NaTa7O19", "0.952"], ["Ta2O5", "0.952"]]),
      { x: 5.3, y: 3.98, w: 2.0, colW: [1.35, 0.65], rowH: 0.34, fontSize: 11.5, border: { type: "none" }, margin: [0, 0.03, 0, 0.03] });
    s.addText([{ text: "LiTaO3", options: { bold: true, fontSize: 15, breakLine: true } }, { text: "new · from MP", options: { fontSize: 9.5 } }], {
      shape: pres.shapes.OVAL, x: cx - 0.68, y: cy - 0.52, w: 1.36, h: 1.04, fill: { color: C.accent1 }, line: { color: C.background1, width: 2.5 },
      color: C.background1, align: "center", valign: "middle", margin: 0, objectName: "new-material",
    });
    const tk = [
      ["FaRecycle", D.green, "Hubs are reused", "Identity keys attach LiTaO3 to existing nodes; only Li-O-Ta is new."],
      ["FaSortAmountUp", D.purple, "Ranked by selectivity", "Sharing R3c (2 materials) says more than sharing O (758)."],
      ["FaLink", C.accent1, "Related materials", "Top-5 similar_to on the build-time scale, plus nearest compositions."],
    ];
    for (let i = 0; i < tk.length; i++) {
      const [ic, col, h, b] = tk[i];
      const y = 1.0 + i * 2.0;
      s.addShape(pres.shapes.OVAL, { x: 7.65, y, w: 0.42, h: 0.42, fill: { color: col }, line: { type: "none" }, objectName: nm("icon-bg") });
      s.addImage({ data: await icon(ic), x: 7.75, y: y + 0.1, w: 0.22, h: 0.22, objectName: nm("icon") });
      text(s, h, { x: 8.15, y, w: 1.45, h: 0.42, fontSize: 12.5, bold: true, valign: "middle" });
      text(s, b, { x: 7.65, y: y + 0.5, w: 1.95, h: 1.3, fontSize: 12, color: C.text2 });
    }
    s.addNotes("Worked example from the running system: LiTaO3 is not in the graph, so it is fetched from Materials Project and related. It attaches to existing hubs, ranked by how few materials share them; the only new node is its chemical system. Related materials come from similar_to on the frozen build-time scale and from composition.");
  }

  // ---------- 18. Verified graph statistics ----------
  {
    const s = frame("Results — Verified Graph Statistics");
    text(s, "Edges — 14,686 total", { x: 0.5, y: 1.05, w: 4.2, h: 0.4, fontSize: 17, bold: true, align: "center" });
    text(s, "Application domains", { x: 5.0, y: 1.05, w: 4.5, h: 0.4, fontSize: 17, bold: true, align: "center" });
    const edges = [["similar_to", "3,969"], ["has_oxidation_state", "2,319"], ["contains", "2,269"], ["classified_as", "2,135"], ["suitable_for", "1,506"], ["has_formula_pattern", "794"], ["has_space_group", "794"], ["member_of", "794"], ["crystal_system", "106"]];
    const doms = [["semiconductor_device", "474"], ["magnetic_material", "328"], ["photocatalyst", "249"], ["battery_cathode", "237"], ["dielectric", "218"]];
    table(s, [[th("Relationship"), th("Count", { align: "right" })], ...edges.map(([a, b]) => [mono(a), { text: b, options: { align: "right" } }])],
      { x: 0.5, y: 1.5, w: 4.2, colW: [3.0, 1.2], rowH: 0.32, fontSize: 12.5 });
    table(s, [[th("Domain"), th("Materials", { align: "right" })], ...doms.map(([a, b]) => [mono(a), { text: b, options: { align: "right" } }]),
      [{ text: "Total", options: { bold: true } }, { text: "1,506", options: { bold: true, align: "right" } }]],
      { x: 5.0, y: 1.5, w: 4.5, colW: [3.1, 1.4], rowH: 0.32, fontSize: 12.5 });
    text(s, "Counts read directly out of the graph store on every build — not estimated. Domain totals match the `suitable_for` edge count exactly; a material can belong to more than one domain, and each edge stores the rule that qualified it.",
      { x: 0.5, y: 5.0, w: 9.0, h: 1.2, fontSize: 15 });
  }

  // ---------- 19. Demo: ask (new) ----------
  {
    const s = frame("Results — Demo: Asking a Question");
    s.addImage({ path: ASSET("ask_results.jpg"), x: 0.4, y: 0.98, w: 6.0, h: 3.83, objectName: "ask-screenshot" });
    s.addImage({ path: ASSET("ask_explanation_short.jpg"), x: 0.4, y: 4.95, w: 4.65, h: 2.01, objectName: "explanation-screenshot" });
    const ann = [
      [C.accent1, "Understood as:", "parsed constraints, including “no Co”."],
      [D.green, "Candidates:", "ranked with a score breakdown."],
      [D.purple, "Evidence path:", "the graph edges behind the top result."],
    ];
    ann.forEach(([col, h, b], i) => {
      const y = 1.05 + i * 1.2;
      numDot(s, i + 1, 6.6, y, 0.38, col, 12);
      text(s, [{ text: h + " ", options: { bold: true } }, { text: b, options: { color: C.text2 } }], { x: 7.1, y: y - 0.05, w: 2.5, h: 1.05, fontSize: 13 });
    });
    numDot(s, 4, 5.25, 5.0, 0.38, D.orange, 12);
    text(s, [{ text: "Grounded explanation: ", options: { bold: true } }, { text: "every claim cites a fact id; the checker verified 69 / 69 claims.", options: { color: C.text2 } }],
      { x: 5.75, y: 4.95, w: 3.85, h: 1.0, fontSize: 13 });
    text(s, "Captured with the rule-based parser and templated explanation (LLM off); retrieval and grounding are the same path.", { x: 5.25, y: 6.05, w: 4.35, h: 0.9, fontSize: 11, italic: true, color: C.text2 });
  }

  // ---------- 20. Demo: add material (new) ----------
  {
    const s = frame("Results — Demo: Adding a New Material");
    s.addImage({ path: ASSET("add_connect_graph.jpg"), x: 0.4, y: 0.98, w: 4.2, h: 3.65, objectName: "connect-graph-screenshot" });
    s.addImage({ path: ASSET("add_shared_nodes.jpg"), x: 4.8, y: 0.98, w: 4.8, h: 3.88, objectName: "shared-nodes-screenshot" });
    s.addImage({ path: ASSET("add_related.jpg"), x: 0.4, y: 4.78, w: 2.2, h: 2.18, objectName: "related-materials-screenshot" });
    const ann = [
      [D.green, "How it connects:", "LiTaO3 fetched from Materials Project and previewed in memory until saved."],
      [D.purple, "Shared graph nodes:", "most selective first; ChemicalSystem Li-O-Ta is the only new node."],
      [C.accent1, "Related materials:", "similar_to by property cosine, plus compositional neighbours."],
    ];
    ann.forEach(([col, h, b], i) => {
      const y = 5.0 + i * 0.68;
      numDot(s, i + 1, 2.85, y + 0.04, 0.36, col, 12);
      text(s, [{ text: h + " ", options: { bold: true } }, { text: b, options: { color: C.text2 } }], { x: 3.35, y, w: 6.25, h: 0.62, fontSize: 12.5, valign: "middle" });
    });
  }

  // ---------- 21. Demo: explorer (new) ----------
  {
    const s = frame("Results — Demo: Graph Explorer Import & Match");
    s.addImage({ path: ASSET("explorer_match_start.jpg"), x: 0.4, y: 1.0, w: 4.5, h: 2.81, objectName: "explorer-wiring" });
    s.addImage({ path: ASSET("explorer_match.jpg"), x: 5.1, y: 1.0, w: 4.5, h: 2.81, objectName: "explorer-settled" });
    text(s, "≈ 5 s — wiring to hubs", { x: 0.4, y: 3.88, w: 4.5, h: 0.3, fontSize: 11.5, italic: true, color: C.text2, align: "center" });
    text(s, "≈ 20 s — settled into the graph", { x: 5.1, y: 3.88, w: 4.5, h: 0.3, fontSize: 11.5, italic: true, color: C.text2, align: "center" });
    const steps = [
      [D.green, "Look up", "NaFePO4 fetched from Materials Project (a CIF would be predicted with CGCNN)."],
      [D.purple, "Match", "Compared with same-formula materials: no polymorph in the graph yet."],
      [D.orange, "Wire", "New node linked to each shared hub, rarest first: ABCD4 (6) … O (760)."],
      [C.accent1, "Settle", "similar_to neighbours added; forces pull the node into place."],
    ];
    steps.forEach(([col, h, b], i) => {
      const x = 0.4 + (i % 2) * 4.7, y = 4.4 + Math.floor(i / 2) * 1.3;
      numDot(s, i + 1, x, y, 0.38, col, 12);
      text(s, h, { x: x + 0.5, y, w: 3.9, h: 0.38, fontSize: 14, bold: true, valign: "middle" });
      text(s, b, { x: x + 0.5, y: y + 0.42, w: 3.9, h: 0.78, fontSize: 12.5, color: C.text2 });
    });
  }

  // ---------- 22. Conclusion (new) ----------
  {
    const s = frame("Conclusion and Future Work");
    bullets(s, [
      { h: "Achieved:", space: 0 },
      "794-material knowledge graph with 14,686 edges, built with **zero LLM calls**.",
      "Every claim in an explanation cited and machine-checked.",
      "New materials connect through shared hubs and `similar_to` edges.",
      "Crystal structure encoding: pre-trained CGCNN predicts properties for uploaded CIF / POSCAR files, added to the graph as labelled predictions.",
      { h: "Future work:", space: 10 },
      "Element substitution (`substitutable_with`).",
      "Neo4j migration (a configuration swap) and a Next.js frontend with CI/CD.",
    ], { x: 0.45, y: 1.0, w: 9.1, h: 6.0, fontSize: 20, valign: "middle", gap: 8 });
  }

  // ---------- 23. References ----------
  {
    const s = frame("References");
    const refs = [
      "Y. Chiang, et al., “LLaMP: Large Language Model Made Powerful for High-Fidelity Materials Knowledge Retrieval and Distillation,” 2024.",
      "H. Zhang, et al., “HoneyComb: A Flexible LLM-Based Agent System for Materials Science,” 2024.",
      "V. Mishra, et al., “LLaMat: Foundational Large Language Models for Materials Research,” 2024.",
      "D. Edge, et al., “From Local to Global: A Graph RAG Approach to Query-Focused Summarization,” Microsoft Research, 2024.",
      "T. Xie and J. C. Grossman, “Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties,” Phys. Rev. Lett., vol. 120, 145301, 2018.",
      "K. Choudhary and B. DeCost, “Atomistic Line Graph Neural Network for improved materials property predictions,” npj Comput. Mater., vol. 7, 185, 2021.",
      "A. Jain, et al., “The Materials Project: A materials genome approach to accelerating materials innovation,” APL Materials, vol. 1, no. 1, 011002, 2013.",
      "S. P. Ong, et al., “Python Materials Genomics (pymatgen),” Comput. Mater. Sci., vol. 68, pp. 314–319, 2013.",
    ];
    const ic = await icon("FaRegFileAlt", "#" + HEX.accent1);
    let y = 1.0;
    for (const r of refs) {
      const h = Math.ceil(r.length / 98) * 0.25 + 0.05;
      s.addImage({ data: ic, x: 0.45, y: y + 0.04, w: 0.22, h: 0.22, objectName: nm("ref-icon") });
      text(s, r, { x: 0.8, y, w: 8.8, h, fontSize: 13, color: C.accent1 });
      y += h + 0.2;
    }
  }

  await pres.writeFile({ fileName: OUT });
  await applyThemeColours(OUT, THEME);
  await fixParagraphs(OUT, { bulletColour: "accent1" }); // Beamer draws item balls in the structure colour
  console.log("wrote", OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
