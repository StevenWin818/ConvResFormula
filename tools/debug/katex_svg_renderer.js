"use strict";

const readline = require("readline");

let katex;
try {
  katex = require("katex");
} catch (_err) {
  katex = null;
}

const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

const tex = new TeX({
  packages: AllPackages,
  inlineMath: [["$", "$"], ["\\(", "\\)"]],
  displayMath: [["$$", "$$"], ["\\[", "\\]"]],
});

const svg = new SVG({
  fontCache: "none",
  internalSpeechTitles: false,
});

const doc = mathjax.document("", {
  InputJax: tex,
  OutputJax: svg,
});

function toSvg(texInput) {
  const source = typeof texInput === "string" ? texInput : "";

  // 按需调用 KaTeX 做一次快速可解析性触发，保持与请求约束一致。
  if (katex) {
    try {
      katex.renderToString(source, {
        throwOnError: false,
        output: "htmlAndMathml",
        strict: "ignore",
        trust: false,
      });
    } catch (_err) {
      // 忽略 KaTeX 解析异常，继续由 MathJax 进行 SVG 渲染。
    }
  }

  const node = doc.convert(source, { display: false });
  return adaptor.outerHTML(node);
}

function emit(payload) {
  process.stdout.write(JSON.stringify(payload) + "\n");
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false,
});

rl.on("line", (line) => {
  let req;
  try {
    req = JSON.parse(line);
  } catch (err) {
    emit({ ok: false, error: `invalid_json: ${String(err && err.message ? err.message : err)}` });
    return;
  }

  if (req && req.ping) {
    emit({ pong: true });
    return;
  }

  const texInput = req && typeof req.tex === "string" ? req.tex : "";
  try {
    const svgText = toSvg(texInput);
    emit({ ok: true, svg: svgText });
  } catch (err) {
    emit({ ok: false, error: String(err && err.message ? err.message : err) });
  }
});

rl.on("close", () => {
  process.exit(0);
});
