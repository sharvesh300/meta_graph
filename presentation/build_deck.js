// Builds presentation/MatGraphRAG.pptx.
// Needs pptxgenjs, jszip, react, react-dom, react-icons and sharp on NODE_PATH.
// Usage: NODE_PATH=<node_modules> node presentation/build_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { fixParagraphs, applyThemeColours } = require("./pptx_fixups");

const HERE = __dirname;
const ASSET = (f) => path.join(HERE, "assets", f);
const OUT = path.join(HERE, "MatGraphRAG.pptx");

const THEME = {
  name: "MatGraphRAG",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B0C11", // ink / dark slide background (matches the explorer canvas)
    lt1: "FFFFFF",
    dk2: "5B6278", // muted text
    lt2: "F1F3F9", // panel tint
    accent1: "4C5FD5", // materials (indigo)
    accent2: "0CA678", // elements (green)
    accent3: "E8890C", // classes / predicted (amber)
    accent4: "1098AD", // oxidation states (teal)
    accent5: "9C36B5", // structure (purple)
    accent6: "D6336C", // application domains (pink)
    hlink: "4C5FD5",
    folHlink: "9C36B5",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "MatGraphRAG";
pres.subject = "Explainable graph-based retrieval for materials discovery";
const C = pres.SchemeColor;
const ACC = [C.accent1, C.accent2, C.accent3, C.accent4, C.accent5, C.accent6];

// ---------- icons ----------
async function icon(name, color = "#FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ---------- layouts ----------
const W = 13.333;
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 6.3, h: 1.1, fontSize: 54, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 2.8, w: 6.0, h: 1.3, fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "meta", type: "body", x: 0.6, y: 4.6, w: 6.0, h: 1.9, fontSize: 15, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 0.3, w: 9, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 2, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.6, w: 12.1, h: 0.75, fontSize: 32, bold: true, color: C.text1, margin: 0, valign: "middle", align: "left" }, text: "" } },
    { text: { text: "MatGraphRAG", options: { x: 0.6, y: 7.0, w: 4, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.5, w: 12.1, h: 0.8, fontSize: 36, bold: true, color: C.background1, margin: 0, valign: "middle", align: "left" }, text: "" } },
  ],
});

// ---------- helpers ----------
let objN = 0;
const nm = (s) => `${s}-${++objN}`;

function content(section, kicker, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  return s;
}

function text(s, t, o) {
  s.addText(t, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", objectName: nm("text") }, o));
}

function card(s, x, y, w, h, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: o.radius ?? 0.12, objectName: nm(o.name || "card"),
    fill: { color: o.fill || C.background2, transparency: o.transparency || 0 },
    line: o.line ? { color: o.line, width: o.lineWidth || 1, dashType: o.dash || "solid" } : { type: "none" },
  });
}

function dot(s, x, y, d, color, o = {}) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { type: "none" }, objectName: nm(o.name || "node") });
}

async function iconDot(s, name, x, y, d, color) {
  dot(s, x, y, d, color, { name: "icon-bg" });
  const pad = d * 0.24;
  s.addImage({ data: await icon(name), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad, objectName: nm("icon") });
}

function numDot(s, n, x, y, d, color, size = 18) {
  s.addText(String(n), {
    shape: pres.shapes.OVAL, x, y, w: d, h: d, fill: { color }, line: { type: "none" },
    fontSize: size, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, objectName: nm("num"),
  });
}

function line(s, x1, y1, x2, y2, o = {}) {
  s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, objectName: nm("edge"),
    line: { color: o.color || C.text2, width: o.width || 1.5, dashType: o.dash || "solid", endArrowType: o.arrow === false ? undefined : "triangle" },
  });
}

function chevron(s, x, y, w, h, color) {
  s.addShape(pres.shapes.CHEVRON, { x, y, w, h, fill: { color }, line: { type: "none" }, objectName: nm("arrow") });
}

function pill(s, label, x, y, w, h, color, o = {}) {
  s.addText(label, {
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: h / 2, x, y, w, h,
    fill: { color, transparency: o.transparency || 0 }, line: o.line ? { color: o.line, width: 1.25, dashType: o.dash || "solid" } : { type: "none" },
    fontSize: o.fontSize || 13, bold: o.bold ?? true, color: o.color || C.background1, align: "center", valign: "middle", margin: 0,
    objectName: nm(o.name || "pill"),
  });
}

function stat(s, big, label, x, y, w, color, o = {}) {
  text(s, big, { x, y, w, h: o.bigH || 0.6, fontSize: o.bigSize || 32, bold: true, color, valign: "bottom" });
  text(s, label, { x, y: y + (o.bigH || 0.6) + 0.05, w, h: o.labelH || 0.55, fontSize: o.labelSize || 13, color: C.text2 });
}

// ======================================================================
async function build() {
  // ---------- 1. Title ----------
  pres.addSection({ title: "Introduction" });
  {
    const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
    s.addImage({ path: ASSET("graph_hero.jpg"), x: 7.05, y: 0.75, w: 5.85, h: 5.61, objectName: "hero-graph" });
    s.addText("MatGraphRAG", { placeholder: "title" });
    s.addText("An explainable graph-based retrieval-augmented reasoning system for intelligent materials discovery", { placeholder: "subtitle" });
    s.addText([
      { text: "[Your name] · [Register / roll number]", options: { breakLine: true, bold: true } },
      { text: "Guide: [Name, designation]", options: { breakLine: true } },
      { text: "[Department, institution]", options: { breakLine: true } },
      { text: "[Review / event] · [Date]" },
    ], { placeholder: "meta", paraSpaceAfter: 4 });
    s.addNotes("Introduce MatGraphRAG: a knowledge graph built from Materials Project data, with graph-based retrieval and evidence-backed explanations on top. The image on the right is the live graph explorer showing all 794 materials and their shared hub nodes.");
  }

  // ---------- 2. Problem statement ----------
  {
    const s = content("Introduction", "PROBLEM STATEMENT", "Materials data is searchable, not reasoned over");
    card(s, 0.6, 1.6, 12.1, 1.05, { name: "question-callout" });
    await iconDot(s, "FaQuoteLeft", 0.85, 1.8, 0.65, C.accent1);
    text(s, [
      { text: "“Find stable, cobalt-free Li cathode materials with a band gap above 2 eV”", options: { italic: true, bold: true, breakLine: true } },
      { text: "…and show exactly why each one qualifies. Today no tool answers this exactly, explains it, and lets you verify it.", options: { color: C.text2 } },
    ], { x: 1.75, y: 1.72, w: 10.75, h: 0.85, fontSize: 17, valign: "middle" });

    const cards = [
      ["FaTable", C.accent2, "Flat records", "Materials Project indexes 154,377 materials as isolated rows. Shared elements, structure families and ionic chemistry stay implicit."],
      ["FaSearch", C.accent5, "Look-alikes, not answers", "Vector RAG ranks by text similarity: 69.6% of its top-10 results violate the query’s hard constraints."],
      ["FaRobot", C.accent6, "Untraceable LLM answers", "Generated recommendations can invent values, and a researcher cannot tell which claim came from where."],
      ["FaPlusCircle", C.accent3, "New materials in isolation", "A new formula or crystal structure has no context: what is it related to, and why?"],
    ];
    const cw = 2.8, gap = 0.3, y = 3.0, h = 3.7;
    for (let i = 0; i < cards.length; i++) {
      const [ic, col, head, body] = cards[i];
      const x = 0.6 + i * (cw + gap);
      card(s, x, y, cw, h, { name: "problem-card" });
      await iconDot(s, ic, x + 0.25, y + 0.3, 0.7, col);
      text(s, head, { x: x + 0.25, y: y + 1.2, w: cw - 0.5, h: 0.65, fontSize: 18, bold: true, valign: "top" });
      text(s, body, { x: x + 0.25, y: y + 1.9, w: cw - 0.5, h: 1.65, fontSize: 14, color: C.text2 });
    }
    s.addNotes("The core problem. Materials databases are large but flat. Similarity search returns look-alikes rather than materials that satisfy numeric constraints (69.6% violations for vector RAG on our benchmark). Plain LLM answers are not traceable. And a new material cannot be placed in the context of known ones.");
  }

  // ---------- 3. Objectives ----------
  {
    const s = content("Introduction", "OBJECTIVES", "Four objectives");
    const obj = [
      [C.accent2, "Build a materials knowledge graph", "From Materials Project data, with deterministic rules and no LLM, reproducible from a local cache."],
      [C.accent1, "Answer questions by graph retrieval", "Turn natural language into typed constraints, then enforce numeric ranges and exclusions exactly on the graph."],
      [C.accent4, "Explain with verified evidence", "Every claim cites a graph edge or a Materials Project record, and a deterministic checker verifies each citation."],
      [C.accent3, "Connect new materials", "Add a material by MP id, formula or crystal structure and link it to its related materials and elements."],
    ];
    const cw = 5.9, ch = 2.4;
    obj.forEach(([col, head, body], i) => {
      const x = 0.6 + (i % 2) * (cw + 0.3), y = 1.65 + Math.floor(i / 2) * (ch + 0.3);
      card(s, x, y, cw, ch, { name: "objective" });
      numDot(s, i + 1, x + 0.35, y + 0.4, 0.75, col, 22);
      text(s, head, { x: x + 1.35, y: y + 0.4, w: cw - 1.65, h: 0.75, fontSize: 20, bold: true, valign: "middle" });
      text(s, body, { x: x + 1.35, y: y + 1.2, w: cw - 1.65, h: 1.0, fontSize: 15, color: C.text2 });
    });
    s.addNotes("Four objectives: construct the graph deterministically; retrieve by traversal with exact filtering; explain with citations that a checker verifies; and connect new materials — by Materials Project id, formula, or crystal structure — to the existing graph.");
  }

  // ---------- 4. Existing approaches ----------
  {
    const s = content("Introduction", "BACKGROUND", "Why existing retrieval falls short");
    const Y = { text: "✓", options: { color: C.accent2, bold: true, fontSize: 22, align: "center" } };
    const N = { text: "✗", options: { color: C.accent6, bold: true, fontSize: 22, align: "center" } };
    const hdr = (t, hi) => ({ text: t, options: { bold: true, fontSize: 15, align: hi ? "center" : "left", color: hi === 2 ? C.background1 : C.text1, fill: { color: hi === 2 ? C.accent1 : C.background2 } } });
    const rowsData = [
      ["Exact numeric constraints (band gap ≥ 2 eV)", N, N, Y],
      ["Exclusions (no cobalt)", N, N, Y],
      ["Multi-hop relations through shared entities", N, Y, Y],
      ["Claims cited and checked against the data", N, N, Y],
      ["New materials placed in the graph", N, N, Y],
      ["Crystal-structure input", N, N, Y],
    ];
    const rows = [[hdr("Capability"), hdr("Vector RAG", 1), hdr("Fixed-hop GraphRAG", 1), hdr("MatGraphRAG", 2)]];
    rowsData.forEach((r) => rows.push([{ text: r[0], options: { fontSize: 15 } }, { ...r[1] }, { ...r[2] }, { ...r[3], options: { ...r[3].options, fill: { color: C.accent1, transparency: 90 } } }]));
    s.addTable(rows, {
      x: 0.6, y: 1.65, w: 12.1, colW: [5.5, 2.2, 2.2, 2.2], rowH: 0.62, valign: "middle", margin: [0, 0.15, 0, 0.15],
      border: { type: "solid", pt: 0.75, color: "D9DCE6" }, fontFace: "Calibri", color: C.text1, objectName: "comparison-table",
    });
    text(s, "Vector RAG: embeddings over material profiles. Fixed-hop GraphRAG: entity seeds plus one-hop expansion, no attribute filtering. Results on slide 13.", { x: 0.6, y: 6.25, w: 12.1, h: 0.55, fontSize: 12, color: C.text2 });
    s.addNotes("Two common baselines. Vector RAG finds text that looks similar but cannot enforce 'band gap above 2 eV' or 'no cobalt'. Fixed-hop GraphRAG reaches the right entities but also cannot apply numeric ranges or exclusions. Neither verifies its claims or places new materials.");
  }

  // ---------- 5. Dataset ----------
  pres.addSection({ title: "Knowledge graph" });
  {
    const s = content("Knowledge graph", "DATASET", "794 materials from eight chemistry-driven clusters");
    const funnel = [
      ["154,377", "materials in Materials Project", 4.6, C.background2, C.text1],
      ["8 clusters", "declarative queries in materials.yaml", 4.1, C.background2, C.text1],
      ["~2,000", "raw candidates, sorted by stability", 3.6, C.background2, C.text1],
      ["794", "unique materials after deduplication", 3.1, C.accent1, C.background1],
    ];
    funnel.forEach(([big, lab, w, fill, col], i) => {
      const y = 1.65 + i * 1.05, x = 0.6 + (4.6 - w) / 2;
      card(s, x, y, w, 0.88, { fill, name: "funnel-step" });
      text(s, [
        { text: big, options: { fontSize: 22, bold: true, breakLine: true } },
        { text: lab, options: { fontSize: 12 } },
      ], { x: x + 0.1, y: y + 0.06, w: w - 0.2, h: 0.76, color: col, align: "center", valign: "middle" });
    });

    text(s, "Clusters", { x: 5.6, y: 1.6, w: 7.1, h: 0.35, fontSize: 16, bold: true });
    const clusters = [
      [C.accent1, "Li–TM oxides", "layered / spinel battery cathodes"],
      [C.accent1, "Li–TM phosphates", "olivine battery cathodes"],
      [C.accent4, "III–V semiconductors", "{B, Al, Ga, In} × {N, P, As, Sb}"],
      [C.accent4, "II–VI semiconductors", "{Zn, Cd, Hg} × {O, S, Se, Te}"],
      [C.accent4, "Group IV semiconductors", "Si, Ge, C, Sn and binaries"],
      [C.accent2, "Binary oxides", "wide-gap and functional oxides"],
      [C.accent5, "ABO₃ perovskites", "dielectric / ferroelectric"],
      [C.accent6, "AB₂O₄ spinels", "magnetic oxides"],
    ];
    clusters.forEach(([col, head, sub], i) => {
      const x = 5.6 + (i % 2) * 3.7, y = 2.05 + Math.floor(i / 2) * 0.88;
      card(s, x, y, 3.4, 0.74, { name: "cluster" });
      dot(s, x + 0.2, y + 0.25, 0.24, col);
      text(s, [
        { text: head, options: { bold: true, fontSize: 14, breakLine: true } },
        { text: sub, options: { fontSize: 12, color: C.text2 } },
      ], { x: x + 0.6, y: y + 0.06, w: 2.7, h: 0.62, valign: "middle" });
    });

    const cov = [["100%", "core properties"], ["93.1%", "oxidation states"], ["19.9%", "dielectric data"], ["18.0%", "elastic data"]];
    text(s, "Property coverage", { x: 5.6, y: 5.6, w: 7.1, h: 0.3, fontSize: 13, bold: true, color: C.text2 });
    cov.forEach(([big, lab], i) => {
      const x = 5.6 + i * 1.8;
      text(s, [{ text: big, options: { fontSize: 22, bold: true, color: C.accent1, breakLine: true } }, { text: lab, options: { fontSize: 12, color: C.text2 } }], { x, y: 5.95, w: 1.75, h: 0.8 });
    });
    s.addNotes("Using all 154k materials would make pairwise similarity intractable and produce a sparse graph, so the dataset is assembled from eight chemistry-driven clusters defined in config/materials.yaml. Results are deduplicated by formula across clusters, keeping the most stable polymorph: about 2,000 candidates become 794 unique materials. Core properties cover all of them; elastic and dielectric data cover under 20%.");
  }

  // ---------- 6. System architecture ----------
  pres.addSection({ title: "Architecture" });
  {
    const s = content("Architecture", "ARCHITECTURE", "System architecture");
    const cols = [
      [C.accent2, "Data sources", [
        ["Materials Project API", "summary + robocrys endpoints"],
        ["Cached raw JSON", "794 records, reproducible builds"],
        ["User input", "MP id · formula · CIF / POSCAR"],
      ]],
      [C.accent1, "Knowledge graph", [
        ["Build pipeline", "fetch → enrich (rules) → write → verify"],
        ["Cognee DataPoints", "identity-keyed nodes and edges"],
        ["Kuzu + LanceDB", "graph store and vector store"],
        ["BGE-M3 embeddings", "local Ollama, 1024-d"],
      ]],
      [C.accent4, "Reasoning services", [
        ["Query parser", "LLM (OpenRouter) + rule fallback"],
        ["Graph retriever", "traverse, filter exactly, rank"],
        ["Evidence → explain → ground", "cited claims, checked"],
        ["Material ingestion", "MP · composition · pretrained CGCNN"],
      ]],
      [C.accent5, "Interfaces", [
        ["FastAPI service", "/query · /materials/ingest · /graph"],
        ["Dashboard", "ask, add material, from structure"],
        ["Graph explorer", "animated import & match"],
        ["CLI scripts", "build, ask, add, evaluate"],
      ]],
    ];
    const cw = 2.75, gap = 0.37;
    cols.forEach(([col, head, items], i) => {
      const x = 0.6 + i * (cw + gap);
      pill(s, head, x, 1.6, cw, 0.5, col, { fontSize: 15, name: "layer-header" });
      items.forEach(([t, sub], j) => {
        const y = 2.3 + j * 1.13;
        card(s, x, y, cw, 0.98, { fill: C.background1, line: col, name: "component" });
        text(s, [{ text: t, options: { bold: true, fontSize: 14, breakLine: true } }, { text: sub, options: { fontSize: 12, color: C.text2 } }],
          { x: x + 0.15, y: y + 0.08, w: cw - 0.3, h: 0.82, valign: "middle" });
      });
      if (i < cols.length - 1) chevron(s, x + cw + 0.08, 4.05, 0.22, 0.4, C.text2);
    });
    s.addNotes("Four layers. Data comes from the Materials Project API and is cached as raw JSON; users can also supply MP ids, formulas or crystal structures. The knowledge graph layer builds and stores the graph with Cognee over Kuzu and LanceDB, with BGE-M3 embeddings from a local Ollama. Reasoning services parse questions, retrieve and rank, build evidence, explain and ground; the ingestion service adds new materials, using a pretrained CGCNN when only a structure is given. Everything is exposed through FastAPI, a dashboard, a graph explorer and CLI scripts.");
  }

  // ---------- 7. KG schema ----------
  {
    const s = content("Architecture", "KNOWLEDGE GRAPH", "Graph schema: 9 node types, 9 edge types");
    const center = { x: 4.3, y: 4.2 };
    // [label, x, y, width, colour, relation, label box [x, y, w, align]]
    const nodes = [
      ["Oxidation state · 131", 1.6, 2.2, 2.15, C.accent4, "has_oxidation_state", [0.6, 2.9, 1.95, "right"]],
      ["Chemical system · 143", 4.3, 1.85, 2.15, C.accent2, "member_of", [4.42, 2.65, 1.2, "left"]],
      ["Formula pattern · 261", 7.0, 2.2, 2.15, C.accent5, "has_formula_pattern", [6.08, 2.9, 1.9, "left"]],
      ["Element · 46", 1.3, 4.2, 1.7, C.accent2, "contains", [2.15, 3.85, 1.15, "center"]],
      ["Space group · 106", 7.6, 4.2, 2.15, C.accent5, "has_space_group", [5.3, 3.85, 1.22, "center"]],
      ["Property class · 11", 1.6, 6.25, 2.15, C.accent3, "classified_as", [0.6, 5.28, 1.95, "right"]],
      ["Application domain · 5", 4.3, 6.4, 2.25, C.accent6, "suitable_for + rule", [4.42, 5.2, 1.6, "left"]],
      ["Material", 7.0, 6.4, 2.15, C.accent1, "similar_to (weight)", [4.45, 5.72, 1.7, "right"]],
    ];
    const NH = 0.5;
    for (const [, x, y] of nodes) line(s, center.x, center.y, x, y, { color: C.text2, width: 1.25, arrow: false });
    line(s, 7.6, 4.2, 7.6, 5.3, { color: C.text2, width: 1.25, arrow: false });
    for (const [lab, x, y, w, col, rel, [lx, ly, lw, al]] of nodes) {
      pill(s, lab, x - w / 2, y - NH / 2, w, NH, col, { fontSize: 13, name: "node-type" });
      text(s, rel, { x: lx, y: ly, w: lw, h: 0.28, fontSize: 11, color: C.text2, align: al, valign: "middle" });
    }
    pill(s, "Crystal system · 7", 7.6 - 1.075, 5.3 - NH / 2, 2.15, NH, C.accent5, { fontSize: 13 });
    text(s, "crystal_system", { x: 7.7, y: 4.6, w: 1.1, h: 0.28, fontSize: 11, color: C.text2, valign: "middle" });
    pill(s, "Material · 794", center.x - 1.0, center.y - 0.36, 2.0, 0.72, C.accent1, { fontSize: 17 });

    card(s, 9.0, 1.6, 3.7, 5.2, { name: "principle-card" });
    text(s, "Design principle", { x: 9.25, y: 1.8, w: 3.2, h: 0.4, fontSize: 18, bold: true });
    const pr = [
      ["FaSlidersH", C.accent1, "Values stay attributes", "Exact band gap, density and moduli live on the Material node, for filtering."],
      ["FaProjectDiagram", C.accent2, "Groups become nodes", "Anything used to group or traverse (element, space group, class…) is a shared hub."],
      ["FaBalanceScale", C.accent3, "Relations carry evidence", "similar_to keeps its cosine weight; suitable_for keeps the rule text that fired."],
    ];
    for (let i = 0; i < pr.length; i++) {
      const [ic, col, h, b] = pr[i];
      const y = 2.4 + i * 1.45;
      await iconDot(s, ic, 9.25, y, 0.5, col);
      text(s, h, { x: 9.9, y, w: 2.65, h: 0.4, fontSize: 15, bold: true, valign: "middle" });
      text(s, b, { x: 9.9, y: y + 0.42, w: 2.65, h: 0.95, fontSize: 13, color: C.text2 });
    }
    s.addNotes("Material is the centre of the schema. Continuous values stay as attributes on the Material node so they can be filtered exactly. Anything used for grouping becomes a shared hub node with an identity key — element, oxidation state, chemical system, formula pattern, space group and crystal system, property class and application domain. Material-to-material similarity is a weighted edge, and suitable_for edges store the rule text that fired.");
  }

  // ---------- 8. Construction pipeline ----------
  {
    const s = content("Architecture", "CONSTRUCTION", "Construction pipeline: deterministic, no LLM");
    const steps = [
      [C.accent2, "Fetch", "mp_client.py", ["One query per cluster sub-pattern", "Dedupe by formula across clusters", "Batch-fetch robocrys text"]],
      [C.accent1, "Enrich", "enrich.py", ["Bucket band gap, stability, magnetism", "Fire domain rules, keep the rule text", "Top-5 similar_to on a 10-d z-scored vector"]],
      [C.accent4, "Write", "pipeline.py", ["Cognee DataPoint pipeline", "Identity keys dedupe shared hubs", "Graph and vectors written together"]],
      [C.accent5, "Verify", "verify.py", ["Node and edge counts vs manifest", "Every material has a contains edge", "Distinct vector collections per field"]],
    ];
    const cw = 2.75, gap = 0.37, y = 1.65, h = 4.0;
    steps.forEach(([col, head, mod, bullets], i) => {
      const x = 0.6 + i * (cw + gap);
      card(s, x, y, cw, h, { name: "pipeline-step" });
      numDot(s, i + 1, x + 0.25, y + 0.3, 0.6, col, 18);
      text(s, head, { x: x + 1.0, y: y + 0.28, w: cw - 1.2, h: 0.4, fontSize: 20, bold: true, valign: "middle" });
      text(s, mod, { x: x + 1.0, y: y + 0.66, w: cw - 1.2, h: 0.3, fontSize: 12, color: C.text2, fontFace: "Courier New" });
      text(s, bullets.map((b, k) => ({ text: b, options: { bullet: true, breakLine: k < bullets.length - 1 } })),
        { x: x + 0.25, y: y + 1.2, w: cw - 0.45, h: 2.6, fontSize: 14, paraSpaceAfter: 8 });
      if (i < steps.length - 1) chevron(s, x + cw + 0.08, y + h / 2 - 0.2, 0.22, 0.4, C.text2);
    });
    const st = [["0", "LLM calls"], ["1,504", "nodes"], ["14,686", "edges"], ["794", "cached JSON records"]];
    st.forEach(([b, l], i) => {
      const x = 0.6 + i * 3.12;
      text(s, [{ text: b + "  ", options: { fontSize: 24, bold: true, color: C.accent1 } }, { text: l, options: { fontSize: 14, color: C.text2 } }], { x, y: 5.95, w: 2.9, h: 0.6, valign: "middle" });
    });
    s.addNotes("Four stages. Fetching is separate and cached, because it is the slow, rate-limited part. Enrichment applies deterministic rules: band-gap and stability buckets, application-domain rules that keep their rule text, and top-5 similarity on a z-scored 10-dimensional property vector. Cognee writes graph and vectors in one pass, deduplicating hubs by identity key. Verification checks counts and structure on every build. No LLM is involved anywhere.");
  }

  // ---------- 9. Interconnection ----------
  {
    const s = content("Architecture", "CONNECTIVITY", "Shared hubs make the graph dense");
    s.addImage({ path: ASSET("explorer_full.jpg"), x: 0.6, y: 1.6, w: 7.4, h: 4.625, objectName: "explorer-screenshot" });
    text(s, "Live graph explorer (/explorer): node colour = type; material colour = DFT band gap", { x: 0.6, y: 6.32, w: 7.4, h: 0.3, fontSize: 11, color: C.text2 });
    const facts = [
      ["95%", "of materials contain oxygen — any two oxides are two hops apart", C.accent2],
      ["87%", "share the O²⁻ oxidation state; Li⁺ alone links 227 materials", C.accent4],
      ["44", "materials share the ABC3 pattern with no element in common", C.accent5],
      ["3,969", "weighted similar_to edges: top-5 neighbours, cosine ≥ 0.6", C.accent1],
    ];
    facts.forEach(([b, l, col], i) => {
      const y = 1.6 + i * 1.2;
      stat(s, b, l, 8.45, y, 4.25, col, { bigSize: 30, bigH: 0.5, labelH: 0.6, labelSize: 14 });
    });
    s.addNotes("Because hubs are shared, the graph connects far more than the material count suggests. Oxygen links 95% of materials; the O2- oxidation state links 87%. Formula patterns group by stoichiometric shape, so BaTiO3 and CaHfO3 are neighbours via ABC3 without sharing an element. On top of that, similar_to edges add weighted shortcuts between materials with similar property profiles. No material is an isolated island.");
  }

  // ---------- 10. GraphRAG pipeline ----------
  pres.addSection({ title: "Retrieval and ingestion" });
  {
    const s = content("Retrieval and ingestion", "RETRIEVAL", "GraphRAG pipeline: question to grounded answer");
    card(s, 0.6, 1.55, 12.1, 0.55, { fill: C.text1, name: "question" });
    text(s, [{ text: "Q  ", options: { bold: true, color: C.accent3 } }, { text: "stable, cobalt-free Li cathode materials with a band gap above 2 eV", options: { color: C.background1 } }],
      { x: 0.85, y: 1.55, w: 11.7, h: 0.55, fontSize: 16, valign: "middle" });
    const steps = [
      [C.accent1, "Parse", "LLM with JSON-schema output → typed constraints; rule-based fallback; values checked against graph vocabulary.", "battery_cathode · band_gap ≥ 2 · stable · contains Li · no Co"],
      [C.accent2, "Retrieve", "Seed constraint nodes, traverse back to materials, intersect, enforce numeric ranges, then rank.", "39 of 77 reached materials pass every constraint; top: LiScO2"],
      [C.accent4, "Evidence", "Flatten into atomic facts [E1] [E2] …: an MP field or a graph edge with its rule text.", "[E4] LiScO2 band_gap = 3.904 eV"],
      [C.accent5, "Explain", "LLM writes claims that cite fact ids, or a templated explanation without an LLM.", "“LiScO2 is a stable cathode candidate … [E4][E15]”"],
      [C.accent3, "Ground", "Checker: ids exist, every number matches a cited fact, no unretrieved materials.", "69 / 69 claims grounded (100%)"],
    ];
    const cw = 2.3, gap = 0.15, y = 2.35;
    steps.forEach(([col, head, body, ex], i) => {
      const x = 0.6 + i * (cw + gap);
      pill(s, `${i + 1}  ${head}`, x, y, cw, 0.5, col, { fontSize: 15 });
      text(s, body, { x: x + 0.05, y: y + 0.65, w: cw - 0.1, h: 1.7, fontSize: 14, color: C.text1 });
      card(s, x, y + 2.4, cw, 1.05, { fill: col, transparency: 88, name: "example" });
      text(s, ex, { x: x + 0.12, y: y + 2.45, w: cw - 0.24, h: 0.95, fontSize: 12, color: C.text1, valign: "middle", bold: true });
    });
    card(s, 0.6, 6.05, 12.1, 0.72, { fill: C.accent1, transparency: 90, name: "design-callout" });
    text(s, [
      { text: "The LLM never selects or filters materials. ", options: { bold: true, color: C.accent1 } },
      { text: "Numeric limits, exclusions and rules are enforced exactly on graph attributes and edges.", options: { color: C.text1 } },
    ], { x: 0.85, y: 6.05, w: 11.6, h: 0.72, fontSize: 15, valign: "middle" });
    s.addNotes("Five steps. The LLM only parses the question into typed constraints and phrases the explanation. Retrieval seeds the constraint nodes, traverses back to materials and enforces every numeric limit and exclusion exactly; ranking combines constraint margin, anchor proximity, BGE-M3 semantic similarity and stability. Evidence becomes numbered atomic facts, the explanation cites them, and a deterministic checker verifies every citation and every number. For this query 69 of 69 claims were grounded.");
  }

  // ---------- 11. Material addition architecture ----------
  {
    const s = content("Retrieval and ingestion", "INGESTION", "Adding a new material: ingestion architecture");
    const boxes = [
      [C.accent2, "Input", ["MP id, formula, CIF or POSCAR", "dashboard, API or CLI"]],
      [C.accent3, "Property source", ["MP record → mp", "formula only → derived", "CGCNN → predicted ± error"]],
      [C.accent5, "Deduplicate", ["formula → existing MP node", "structure → StructureMatcher", "no match → new struct-… id"]],
      [C.accent1, "Build nodes", ["same NodeFactory as the build", "identity keys reuse hubs", "creates only missing hubs"]],
      [C.accent4, "Relate", ["hubs ranked by selectivity", "similar_to, frozen z-scale", "compositional neighbours"]],
      [C.accent6, "Relation report", ["properties with provenance", "rules fired / not fired / cannot evaluate"]],
      [C.accent1, "Explain & ground", ["LLM or templated text", "each claim checked against facts"]],
      [C.accent2, "Preview or persist", ["in-memory overlay, queryable", "or saved, replayed on rebuild"]],
    ];
    const cw = 2.75, gap = 0.37, h = 1.9, rows = [1.6, 4.15];
    boxes.forEach(([col, head, lines], i) => {
      const r = Math.floor(i / 4), c = i % 4;
      const x = 0.6 + c * (cw + gap), y = rows[r];
      card(s, x, y, cw, h, { fill: C.background1, line: col, lineWidth: 1.5, name: "ingest-step" });
      numDot(s, i + 1, x + 0.18, y + 0.18, 0.42, col, 13);
      text(s, head, { x: x + 0.7, y: y + 0.15, w: cw - 0.85, h: 0.48, fontSize: 15, bold: true, valign: "middle" });
      text(s, lines.map((l, k) => ({ text: l, options: { bullet: { indent: 11 }, breakLine: k < lines.length - 1 } })),
        { x: x + 0.15, y: y + 0.72, w: cw - 0.25, h: 1.08, fontSize: 13, color: C.text2, paraSpaceAfter: 3 });
      if (c < 3) chevron(s, x + cw + 0.08, y + h / 2 - 0.2, 0.22, 0.4, C.text2);
    });
    // elbow from step 4 down and back to step 5
    const x4 = 0.6 + 3 * (cw + gap) + cw / 2, x5 = 0.6 + cw / 2, yTop = rows[0] + h, yMid = 3.83;
    line(s, x4, yTop, x4, yMid, { arrow: false, color: C.text2 });
    line(s, x4, yMid, x5, yMid, { arrow: false, color: C.text2 });
    line(s, x5, yMid, x5, rows[1], { color: C.text2 });

    text(s, "Provenance labels", { x: 0.6, y: 6.25, w: 2.0, h: 0.42, fontSize: 13, bold: true, color: C.text2, valign: "middle" });
    [["mp", C.accent2, false], ["derived", C.accent1, false], ["guessed", C.accent5, false], ["predicted", C.accent3, true]].forEach(([l, col, dash], i) => {
      pill(s, l, 2.6 + i * 1.6, 6.28, 1.4, 0.36, col, { fontSize: 12, transparency: 85, color: col, line: col, dash: dash ? "dash" : "solid" });
    });
    text(s, "Predicted facts are hedged in explanations and drawn dashed.", { x: 9.1, y: 6.25, w: 3.6, h: 0.42, fontSize: 12, color: C.text2, valign: "middle" });
    s.addNotes("A new material can arrive as a Materials Project id, a formula or a crystal structure. The property source depends on the input: the MP record, only what the formula determines, or a pretrained CGCNN for a bare structure, each labelled with its provenance. Deduplication resolves formulas to existing MP nodes and uses StructureMatcher for structures. The same NodeFactory as the full build creates the nodes, so identity keys attach the material to existing hubs. Relating ranks the shared hubs, finds similar_to and compositional neighbours and evaluates the application rules. The report can be explained and grounded, and the material is either previewed in memory or persisted and replayed on rebuild.");
  }

  // ---------- 12. Edge connections (LiTaO3) ----------
  {
    const s = content("Retrieval and ingestion", "INGESTION", "How a new material connects: LiTaO3");
    const cx = 4.6, cy = 4.22;
    const groups = [
      // [x, y, w, h, color, header, lines, anchor]
      [0.6, 1.6, 2.6, 1.45, C.accent2, "contains → Element", ["Ta · shared with 51", "Li · shared with 245", "O · shared with 758"], [3.2, 2.33]],
      [0.6, 3.5, 2.6, 1.45, C.accent4, "has_oxidation_state", ["Ta⁵⁺ · shared with 38", "Li⁺ · shared with 227", "O²⁻ · shared with 697"], [3.2, 4.22]],
      [0.6, 5.4, 2.6, 1.45, C.accent3, "classified_as / suitable_for", ["wide_gap · 145", "stable · 366", "dielectric · 220 (rule fired)"], [3.2, 6.12]],
      [3.45, 1.6, 2.3, 1.45, C.accent5, "has_space_group", ["R3c (#161) · 2", "→ Trigonal system", ""], [cx, 3.05]],
      [3.45, 5.4, 2.3, 1.45, C.accent5, "pattern / chem. system", ["ABC3 · shared with 45", "Li-O-Ta · new node", ""], [cx, 5.4]],
    ];
    for (const g of groups) {
      const [x, y, w, h, col, , , [ax, ay]] = g;
      line(s, cx, cy, ax, ay, { color: col, width: 1.75 });
    }
    for (const [x, y, w, h, col, head, lines] of groups) {
      card(s, x, y, w, h, { fill: col, transparency: 90, line: col, name: "hub-group" });
      text(s, head, { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: 0.32, fontSize: 13, bold: true, color: col, valign: "middle", fontFace: "Calibri" });
      text(s, lines.filter(Boolean).map((l, k, a) => ({ text: l, options: { breakLine: k < a.length - 1 } })),
        { x: x + 0.15, y: y + 0.47, w: w - 0.3, h: h - 0.55, fontSize: 13, color: C.text1, paraSpaceAfter: 2 });
    }
    // related materials panel
    line(s, cx, cy, 6.0, cy, { color: C.accent1, width: 1.75 });
    card(s, 6.0, 1.6, 2.75, 5.25, { fill: C.accent1, transparency: 90, line: C.accent1, name: "related-materials" });
    text(s, "Related materials", { x: 6.15, y: 1.7, w: 2.45, h: 0.35, fontSize: 14, bold: true, color: C.accent1, valign: "middle" });
    text(s, "similar_to · property cosine", { x: 6.15, y: 2.1, w: 2.45, h: 0.3, fontSize: 12, color: C.text2 });
    const sim = [["Ba3Hf2O7", "0.976"], ["SrTa4O11", "0.967"], ["Na2Ta4O11", "0.956"], ["K2Ta4O11", "0.954"], ["Ba4Hf3O10", "0.953"]];
    const comp = [["Ta3O8", "0.953"], ["NaTa7O19", "0.952"], ["Ta2O5", "0.952"]];
    const rowsT = (arr) => arr.map(([f, w]) => [{ text: f, options: { bold: true } }, { text: w, options: { align: "right", color: C.text2 } }]);
    s.addTable(rowsT(sim), { x: 6.15, y: 2.45, w: 2.45, colW: [1.6, 0.85], rowH: 0.36, fontSize: 13, fontFace: "Calibri", color: C.text1, margin: [0, 0.05, 0, 0.05], valign: "middle", objectName: "similar-table" });
    text(s, "compositional neighbours", { x: 6.15, y: 4.45 + 0.05, w: 2.45, h: 0.3, fontSize: 12, color: C.text2 });
    s.addTable(rowsT(comp), { x: 6.15, y: 4.85, w: 2.45, colW: [1.6, 0.85], rowH: 0.36, fontSize: 13, fontFace: "Calibri", color: C.text1, margin: [0, 0.05, 0, 0.05], valign: "middle", objectName: "composition-table" });

    // centre node on top of the edges
    s.addText([{ text: "LiTaO3", options: { bold: true, fontSize: 18, breakLine: true } }, { text: "new · from MP", options: { fontSize: 11 } }], {
      shape: pres.shapes.OVAL, x: cx - 0.8, y: cy - 0.62, w: 1.6, h: 1.24, fill: { color: C.accent1 }, line: { color: C.background1, width: 3 },
      color: C.background1, align: "center", valign: "middle", margin: 0, objectName: "new-material",
    });

    const tk = [
      ["FaRecycle", C.accent2, "Hubs are reused", "Identity keys attach LiTaO3 to existing nodes; only Li-O-Ta is created."],
      ["FaSortAmountUp", C.accent5, "Ranked by selectivity", "Sharing R3c (2 materials) says more than sharing O (758)."],
      ["FaLink", C.accent1, "Related materials", "Top-5 similar_to on the build-time scale, plus the nearest compositions."],
    ];
    for (let i = 0; i < tk.length; i++) {
      const [ic, col, h, b] = tk[i];
      const y = 1.65 + i * 1.75;
      await iconDot(s, ic, 9.1, y, 0.55, col);
      text(s, h, { x: 9.8, y, w: 2.9, h: 0.55, fontSize: 16, bold: true, valign: "middle" });
      text(s, b, { x: 9.8, y: y + 0.6, w: 2.9, h: 1.0, fontSize: 14, color: C.text2 });
    }
    s.addNotes("A worked example from the live system: LiTaO3 is not in the graph, so it is fetched from Materials Project and related. It attaches to existing element, oxidation-state, space-group and formula-pattern hubs; the only new node is its chemical system Li-O-Ta. Hubs are ranked by selectivity: R3c is shared with only 2 materials, oxygen with 758. Its dielectric rule fires (band gap 3.7 eV, non-magnetic). Related materials come from two sources: similar_to edges using the frozen build-time property scale, and compositional neighbours.");
  }

  // ---------- 13. Results ----------
  pres.addSection({ title: "Results and demo" });
  {
    const s = content("Results and demo", "RESULTS", "Retrieval results: exact answers, zero violations");
    const labels = ["Recall@10", "MRR", "nDCG@10"];
    s.addChart(pres.charts.BAR, [
      { name: "Vector RAG", labels, values: [0.326, 0.386, 0.308] },
      { name: "Fixed-hop GraphRAG", labels, values: [0.613, 0.663, 0.598] },
      { name: "MatGraphRAG", labels, values: [0.926, 0.945, 0.910] },
    ], {
      x: 0.6, y: 1.6, w: 7.3, h: 4.75, barDir: "col", barGapWidthPct: 60,
      chartColors: [HEX.dk2, HEX.accent5, HEX.accent1], chartColorsOpacity: 100,
      showTitle: true, title: "Ranking quality (55 questions, gold constraints)", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
      showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00", valAxisLabelFontSize: 11, valAxisLabelColor: HEX.dk2, valAxisLabelFontFace: "+mn-lt",
      catAxisLabelFontSize: 12, catAxisLabelColor: HEX.dk1, catAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "E3E5EC", size: 0.75 }, catGridLine: { style: "none" }, objectName: "retrieval-chart",
    });
    const st = [
      ["0.910", "nDCG@10, vs 0.308 for vector RAG and 0.598 for fixed-hop GraphRAG", C.accent1],
      ["0%", "top-10 constraint violations, vs 69.6% and 41.8%", C.accent2],
      ["1.000", "nDCG@10 on single, multi-constraint, multi-hop and sparse-data tiers", C.accent5],
    ];
    st.forEach(([b, l, col], i) => {
      const y = 1.65 + i * 1.55;
      card(s, 8.3, y, 4.4, 1.35, { name: "stat-card" });
      text(s, b, { x: 8.5, y: y + 0.12, w: 1.6, h: 1.1, fontSize: 32, bold: true, color: col, valign: "middle" });
      text(s, l, { x: 10.15, y: y + 0.12, w: 2.4, h: 1.1, fontSize: 13, color: C.text2, valign: "middle" });
    });
    text(s, "Benchmark: 55 questions in 5 tiers. Relevance comes from an independent oracle over the raw Materials Project JSON, which imports nothing from the graph build or the retriever.", { x: 0.6, y: 6.4, w: 12.1, h: 0.45, fontSize: 11, color: C.text2 });
    s.addNotes("Evaluated on 55 questions in five tiers, with relevance computed by an independent oracle over the raw Materials Project JSON. With correct constraints, graph retrieval matches the oracle exactly on four of five tiers. nDCG@10 is 0.910 against 0.308 for vector RAG and 0.598 for fixed-hop GraphRAG, and none of our top-10 results violates a constraint, against 69.6% and 41.8% for the baselines.");
  }

  // ---------- 14. Demo: ask ----------
  {
    const s = content("Results and demo", "DEMO", "Demo: asking a question");
    s.addImage({ path: ASSET("ask_results.jpg"), x: 0.6, y: 1.6, w: 7.4, h: 4.73, objectName: "ask-screenshot" });
    s.addImage({ path: ASSET("ask_explanation.jpg"), x: 8.3, y: 1.6, w: 4.43, h: 2.33, objectName: "explanation-screenshot" });
    const ann = [
      [C.accent1, "Understood as", "chips show the parsed constraints, including “no Co”."],
      [C.accent2, "Ranked candidates", "with a score breakdown and the evidence path of the selected one."],
      [C.accent3, "Grounded explanation", "every claim cites a fact id: 69 / 69 grounded."],
    ];
    ann.forEach(([col, h, b], i) => {
      const y = 4.2 + i * 0.75;
      numDot(s, i + 1, 8.3, y + 0.08, 0.42, col, 14);
      text(s, [{ text: h + " ", options: { bold: true } }, { text: b, options: { color: C.text2 } }], { x: 8.9, y, w: 3.83, h: 0.66, fontSize: 13, valign: "middle" });
    });
    s.addNotes("The dashboard answering the cathode question. The parsed constraints are shown as chips, candidates are ranked with their score components, and the evidence path for the selected material is drawn as a small graph. The explanation lists each claim with a tick when the checker grounds it. This capture used the rule-based parser and templated explanation (LLM switched off), which take the same path through retrieval and grounding.");
  }

  // ---------- 15. Demo: add material ----------
  {
    const s = content("Results and demo", "DEMO", "Demo: adding LiTaO3 to the graph");
    const shots = [["add_connect_graph.jpg", 3.97, "connect-graph-screenshot"], ["add_shared_nodes.jpg", 4.26, "shared-nodes-screenshot"], ["add_related.jpg", 3.48, "related-materials-screenshot"]];
    const shotX = [];
    let sx = 0.6;
    for (const [f, w, n] of shots) { s.addImage({ path: ASSET(f), x: sx, y: 1.6, w, h: 3.45, objectName: n }); shotX.push([sx, w]); sx += w + 0.2; }
    const ann = [
      [C.accent2, "How it connects", "Fetched from Materials Project and previewed in memory until saved."],
      [C.accent5, "Shared graph nodes", "Most selective first; ChemicalSystem Li-O-Ta is the only new node."],
      [C.accent1, "Related materials", "similar_to by property cosine, plus compositional neighbours."],
    ];
    ann.forEach(([col, h, b], i) => {
      const [x, aw] = shotX[i];
      numDot(s, i + 1, x, 5.35, 0.42, col, 14);
      text(s, h, { x: x + 0.55, y: 5.35, w: aw - 0.55, h: 0.42, fontSize: 15, bold: true, valign: "middle" });
      text(s, b, { x, y: 5.88, w: aw, h: 0.8, fontSize: 14, color: C.text2 });
    });
    s.addNotes("The Add material tab with LiTaO3. The left panel draws how the material connects: elements, oxidation states, space group, formula pattern, classes, the dielectric domain and its five most similar materials. The tables list the shared nodes ranked by selectivity and the related materials with their scores. Nothing is saved unless 'save to graph' is ticked.");
  }

  // ---------- 16. Demo: explorer ----------
  {
    const s = content("Results and demo", "DEMO", "Demo: import & match in the graph explorer");
    s.addImage({ path: ASSET("explorer_match_start.jpg"), x: 0.6, y: 1.6, w: 5.9, h: 3.69, objectName: "explorer-wiring" });
    s.addImage({ path: ASSET("explorer_match.jpg"), x: 6.8, y: 1.6, w: 5.9, h: 3.69, objectName: "explorer-settled" });
    pill(s, "≈ 5 s · wiring", 0.75, 1.75, 1.6, 0.36, C.accent3, { fontSize: 12 });
    pill(s, "≈ 20 s · settled", 6.95, 1.75, 1.75, 0.36, C.accent1, { fontSize: 12 });
    const steps = [
      [C.accent2, "Look up", "NaFePO4 fetched from MP (a CIF would be predicted with CGCNN)."],
      [C.accent5, "Match", "Compared with same-formula materials: no polymorph in the graph."],
      [C.accent3, "Wire", "New node linked to each shared hub, rarest first: ABCD4 (6) → O (760)."],
      [C.accent1, "Settle", "similar_to neighbours added; forces pull the node into place."],
    ];
    const cw = 2.85, gap = 0.233;
    steps.forEach(([col, h, b], i) => {
      const x = 0.6 + i * (cw + gap);
      numDot(s, i + 1, x, 5.55, 0.42, col, 14);
      text(s, h, { x: x + 0.55, y: 5.55, w: cw - 0.55, h: 0.42, fontSize: 15, bold: true, valign: "middle" });
      text(s, b, { x, y: 6.05, w: cw, h: 0.8, fontSize: 13, color: C.text2 });
    });
    s.addNotes("The graph explorer animates the same ingestion. Here NaFePO4, which is not in the graph, is looked up in Materials Project and compared with same-formula materials. No polymorph matches, so a new node spawns and is wired to each shared hub in turn, most selective first, with similar_to neighbours last; the force layout then pulls it into place. Ticking 'save to graph' persists it through Cognee.");
  }

  // ---------- 17. Closing ----------
  pres.addSection({ title: "Conclusion" });
  {
    const s = pres.addSlide({ masterName: "CLOSING", sectionTitle: "Conclusion" });
    s.addText("Conclusion and next steps", { placeholder: "title" });
    const cols = [
      [C.accent2, "Delivered", [
        "794-material knowledge graph, 14,686 edges, built with zero LLM calls",
        "GraphRAG answers: nDCG@10 0.910 with 0% constraint violations",
        "Every claim cited and machine-checked",
        "New materials connect through shared hubs and similar_to",
      ]],
      [C.accent3, "Limitations", [
        "Similarity queries: nDCG@10 0.517; property similarity is not a structural analogue",
        "LLM question parsing is the end-to-end bottleneck; rules are the fallback",
        "Elastic and dielectric data cover under 20% of materials",
      ]],
      [C.accent4, "Next steps", [
        "Structural embeddings and structurally_similar edges",
        "Element substitution (substitutable_with)",
        "Neo4j migration (a configuration swap)",
        "Next.js frontend, Docker and CI/CD",
      ]],
    ];
    const cw = 3.85, gap = 0.27;
    cols.forEach(([col, head, items], i) => {
      const x = 0.6 + i * (cw + gap);
      card(s, x, 1.6, cw, 4.55, { fill: C.background1, transparency: 93, name: "closing-column" });
      dot(s, x + 0.3, 1.88, 0.3, col);
      text(s, head, { x: x + 0.75, y: 1.8, w: cw - 1.0, h: 0.45, fontSize: 20, bold: true, color: C.background1, valign: "middle" });
      text(s, items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })),
        { x: x + 0.3, y: 2.5, w: cw - 0.55, h: 3.5, fontSize: 15, color: C.background2, paraSpaceAfter: 10 });
    });
    text(s, "Thank you — questions?", { x: 0.6, y: 6.35, w: 12.1, h: 0.6, fontSize: 26, bold: true, color: C.background1, align: "right", valign: "middle" });
    s.addNotes("To conclude: the knowledge graph is built deterministically, retrieval is exact and explanations are verifiable, and new materials are connected through shared hubs and similarity edges. The main open problem is structural similarity, which property-based similar_to does not capture; structural embeddings are the next step, followed by element substitution, Neo4j and deployment.");
  }

  await pres.writeFile({ fileName: OUT });
  await applyThemeColours(OUT, THEME);
  await fixParagraphs(OUT);
  console.log("wrote", OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
