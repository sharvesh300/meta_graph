// Post-processing for decks written by pptxgenjs.
const fs = require("fs");
const JSZip = require("jszip");

// pptxgenjs writes an <a:pPr> before every run of a multi-run paragraph; only the first is valid.
// With `bulletColour`, bullets also get <a:buClr>, which pptxgenjs cannot set.
async function fixParagraphs(file, { bulletColour } = {}) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const pPr = /<a:pPr\b[^>]*\/>|<a:pPr\b[^>]*>.*?<\/a:pPr>/gs;
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    let xml = await zip.file(name).async("string");
    xml = xml.replace(/<a:p>(.*?)<\/a:p>/gs, (whole, inner) => {
      let first = true;
      const kept = inner.replace(pPr, (m, offset) => {
        if (first && offset === 0) { first = false; return m; }
        return "";
      });
      return `<a:p>${kept}</a:p>`;
    });
    if (bulletColour) {
      const clr = `<a:buClr><a:schemeClr val="${bulletColour}"/></a:buClr>`;
      xml = xml.replace(pPr, (m) => (m.includes("<a:buChar") && !m.includes("<a:buClr") ? m.replace(/<a:(buSzPct|buSzPts|buFont|buChar)\b/, clr + "<a:$1") : m));
    }
    zip.file(name, xml);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

// pptxgenjs writes Office's stock colour scheme; scheme colours (pres.SchemeColor) only follow the deck's
// palette once the theme part carries it. Fonts are already written from pres.theme.
async function applyThemeColours(file, theme) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/theme\/theme\d+\.xml$/.test(n))) {
    let xml = await zip.file(name).async("string");
    xml = xml.replace(/<a:clrScheme name="[^"]*">/, `<a:clrScheme name="${esc(theme.name)}">`);
    for (const [slot, hex] of Object.entries(theme.colors)) {
      xml = xml.replace(new RegExp(`<a:${slot}>.*?</a:${slot}>`, "s"), `<a:${slot}><a:srgbClr val="${hex}"/></a:${slot}>`);
    }
    zip.file(name, xml);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

module.exports = { fixParagraphs, applyThemeColours };
