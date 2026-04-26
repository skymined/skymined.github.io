import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import rehypeMathjax from "rehype-mathjax/svg"
//@ts-ignore
import rehypeTypst from "@myriaddreamin/rehype-typst"
import { QuartzTransformerPlugin } from "../types"
import { KatexOptions } from "katex"
import { Options as MathjaxOptions } from "rehype-mathjax/svg"
//@ts-ignore
import { Options as TypstOptions } from "@myriaddreamin/rehype-typst"

interface Options {
  renderEngine: "katex" | "mathjax" | "typst"
  customMacros: MacroType
  katexOptions: Omit<KatexOptions, "macros" | "output">
  mathJaxOptions: Omit<MathjaxOptions, "macros">
  typstOptions: TypstOptions
}

// mathjax macros
export type Args = boolean | number | string | null
interface MacroType {
  [key: string]: string | Args[]
}

const normalizeStandaloneDisplayMath = (src: string) => {
  const lines = src.split(/\r?\n/)
  const normalized: string[] = []
  let activeFence: string | null = null
  let activeDisplayMathPrefix: string | null = null

  for (const line of lines) {
    const trimmed = line.trim()
    const fenceMatch = trimmed.match(/^(```+|~~~+)/)

    if (fenceMatch) {
      const marker = fenceMatch[1]
      if (activeFence === null) {
        activeFence = marker
      } else if (activeFence[0] === marker[0] && marker.length >= activeFence.length) {
        activeFence = null
      }

      normalized.push(line)
      continue
    }

    if (activeFence !== null) {
      normalized.push(line)
      continue
    }

    if (activeDisplayMathPrefix !== null) {
      const closingMatch = line.match(/^(.*)\$\$\s*$/)

      if (closingMatch) {
        const beforeClose = closingMatch[1]
        if (beforeClose.trim().length > 0) {
          normalized.push(beforeClose)
        }

        normalized.push(`${activeDisplayMathPrefix}$$`)
        activeDisplayMathPrefix = null
        continue
      }

      normalized.push(line)
      continue
    }

    const displayMathMatch = line.match(/^(\s*(?:>\s*)*)\$\$(.*)$/)
    if (displayMathMatch) {
      const prefix = displayMathMatch[1]
      const remainder = displayMathMatch[2]
      const closingFenceIndex = remainder.lastIndexOf("$$")
      const hasClosingFence =
        closingFenceIndex >= 0 && remainder.slice(closingFenceIndex + 2).trim().length === 0

      if (hasClosingFence) {
        const expression = remainder.slice(0, closingFenceIndex).trim()
        if (expression.length > 0) {
          normalized.push(`${prefix}$$`, `${prefix}${expression}`, `${prefix}$$`)
          continue
        }
      } else if (remainder.trim().length > 0) {
        normalized.push(`${prefix}$$`, `${prefix}${remainder.trimEnd()}`)
        activeDisplayMathPrefix = prefix
        continue
      }

      normalized.push(line)
      continue
    }

    normalized.push(line)
  }

  return normalized.join("\n")
}

export const Latex: QuartzTransformerPlugin<Partial<Options>> = (opts) => {
  const engine = opts?.renderEngine ?? "katex"
  const macros = opts?.customMacros ?? {}
  return {
    name: "Latex",
    textTransform(_ctx, src) {
      return normalizeStandaloneDisplayMath(src)
    },
    markdownPlugins() {
      return [remarkMath]
    },
    htmlPlugins() {
      switch (engine) {
        case "katex": {
          return [[rehypeKatex, { output: "html", macros, ...(opts?.katexOptions ?? {}) }]]
        }
        case "typst": {
          return [[rehypeTypst, opts?.typstOptions ?? {}]]
        }
        default:
        case "mathjax": {
          return [
            [
              rehypeMathjax,
              {
                ...(opts?.mathJaxOptions ?? {}),
                tex: {
                  ...(opts?.mathJaxOptions?.tex ?? {}),
                  macros,
                },
              },
            ],
          ]
        }
      }
    },
    externalResources() {
      switch (engine) {
        case "katex":
          return {
            css: [{ content: "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" }],
            js: [
              {
                // fix copy behaviour: https://github.com/KaTeX/KaTeX/blob/main/contrib/copy-tex/README.md
                src: "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/copy-tex.min.js",
                loadTime: "afterDOMReady",
                contentType: "external",
              },
            ],
          }
      }
    },
  }
}
